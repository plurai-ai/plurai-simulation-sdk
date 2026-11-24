"""The ``overrides`` of ``ci.yaml``: committed rules that ignore a scenario, a policy or one scenario × policy.

A rule selects by ``scenario``, ``policy`` or both (an exact name or a ``*`` glob; no regex), says ``ignore: true`` and gives
a ``reason``; ``expires`` is an optional date. Ignore is the only action: a tier is changed in the eval.

Precedence for one scenario × policy: a rule naming both beats a scenario rule, which beats a policy rule; within a kind
fewer globs win. Two rules that stay tied and differ in reason or expiry are refused, because picking one would be
silent. :func:`parse_overrides` decides ties from the selectors alone, so ``plurai ci validate`` needs no backend.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from functools import cache
from typing import Final

from plurai.report.run import Cell, Ignored, NotRun, OverrideNote, Run

_KEYS: Final = frozenset({'scenario', 'policy', 'ignore', 'reason', 'expires'})
#: A rule that ends within this many days is a note.
EXPIRY_NOTE_DAYS: Final = 14
#: A name, or a name with ``*`` in it: anything but whitespace and the characters of a regex or of a richer glob.
_SELECTOR: Final = re.compile(r'[^\s?\[\]{}()|^$+\\]+')


class OverridesError(ValueError):
    """``overrides`` cannot be used. A command exits 65 (``REFUSED``) with the message, which names the rule and the fix."""


@dataclass(frozen=True)
class Rule:
    """One rule that ignores what it selects."""

    scenario: str | None
    policy: str | None
    reason: str
    expires: date | None = None

    def describe(self) -> str:
        """The rule as a reader names it: ``scenario legacy_*``, ``policy friendly_tone`` or ``refund_over_limit / never_refund``."""
        if self.scenario is not None and self.policy is not None:
            return f'{self.scenario} / {self.policy}'
        return f'scenario {self.scenario}' if self.scenario is not None else f'policy {self.policy}'

    @property
    def rank(self) -> tuple[int, int]:
        """What beats what: lower wins. The kind first (both selectors, scenario only, policy only), then the globs."""
        kind = 0 if self.scenario is not None and self.policy is not None else 1 if self.scenario is not None else 2
        return (kind, sum('*' in part for part in (self.scenario, self.policy) if part is not None))

    def matches(self, scenario: str, policy: str) -> bool:
        return (self.scenario is None or glob_match(self.scenario, scenario)) and (self.policy is None or glob_match(self.policy, policy))

    def active_on(self, today: date) -> bool:
        """Whether the rule applies: through its ``expires`` date, and not after."""
        return self.expires is None or today <= self.expires


def glob_match(pattern: str, name: str) -> bool:
    """Whether ``name`` matches ``pattern``, where ``*`` stands for any run of characters and nothing else is special."""
    return re.fullmatch('.*'.join(re.escape(part) for part in pattern.split('*')), name, re.DOTALL) is not None


def globs_overlap(left: str, right: str) -> bool:
    """Whether some name matches both patterns."""

    @cache
    def meet(i: int, j: int) -> bool:
        if i == len(left) and j == len(right):
            return True
        if i < len(left) and left[i] == '*':
            if meet(i + 1, j) or (j < len(right) and right[j] != '*' and meet(i, j + 1)):
                return True
        if j < len(right) and right[j] == '*':
            if meet(i, j + 1) or (i < len(left) and left[i] != '*' and meet(i + 1, j)):
                return True
        return i < len(left) and j < len(right) and left[i] == right[j] != '*' and meet(i + 1, j + 1)

    return meet(0, 0)


def _selectors_overlap(a: Rule, b: Rule) -> bool:
    """Whether some scenario × policy is selected by both rules. A selector a rule leaves out matches anything."""
    scenario = a.scenario is None or b.scenario is None or globs_overlap(a.scenario, b.scenario)
    policy = a.policy is None or b.policy is None or globs_overlap(a.policy, b.policy)
    return scenario and policy


def _quoted(rule: Rule) -> str:
    until = f', expires {rule.expires.isoformat()}' if rule.expires else ''
    return f'{rule.describe()} ("{rule.reason}"{until})'


def _selector(value: object, key: str, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise OverridesError(f'{where}: `{key}` is a name or a `*` glob, not {value!r}')
    if not _SELECTOR.fullmatch(value):
        what = 'no regex' if re.search(r'[\^$()|+\\]', value) else 'only `*` is a wildcard'
        raise OverridesError(f'{where}: `{key}: {value}` is not a name or a `*` glob ({what})')
    return value


def _expires(value: object, where: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not hasattr(value, 'hour'):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise OverridesError(f'{where}: `expires` is a date such as 2026-12-01, not {value!r}')


def _rule(raw: object, index: int) -> Rule:
    where = f'overrides[{index}]'
    if not isinstance(raw, Mapping):
        raise OverridesError(f'{where} is not a mapping; a rule has `scenario` and/or `policy`, `ignore: true` and a `reason`')
    for key in raw:
        if key not in _KEYS:
            hint = ' (a tier is changed in the eval with /edit-evals; `ignore: true` is the only action)' if key == 'tier' else ''
            raise OverridesError(f'{where}: `{key}` is not a key of an override{hint}; the keys are scenario, policy, ignore, reason, expires')
    scenario = _selector(raw.get('scenario'), 'scenario', where)
    policy = _selector(raw.get('policy'), 'policy', where)
    if scenario is None and policy is None:
        raise OverridesError(f'{where}: name a `scenario`, a `policy` or both')
    if raw.get('ignore') is not True:
        raise OverridesError(f'{where}: `ignore: true` is the only action an override has')
    reason = raw.get('reason')
    if not isinstance(reason, str) or not reason.strip():
        raise OverridesError(f'{where}: a `reason` is required')
    return Rule(scenario, policy, ' '.join(reason.split()), _expires(raw.get('expires'), where))


def parse_overrides(raw: object) -> tuple[Rule, ...]:
    """The rules of an ``overrides`` value, in order. Raises :class:`OverridesError` for a malformed rule or for two rules
    that tie and disagree; the message names both rules."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise OverridesError('overrides is not a list of rules')
    rules = tuple(_rule(item, index) for index, item in enumerate(raw))
    for later, rule in enumerate(rules):
        for earlier in rules[:later]:
            if earlier.rank == rule.rank and (earlier.reason, earlier.expires) != (rule.reason, rule.expires) and _selectors_overlap(earlier, rule):
                raise OverridesError(
                    f'overrides tie: {_quoted(earlier)} and {_quoted(rule)} can select the same scenario × policy at the same '
                    'precedence and disagree. Name one more exactly or give both the same reason and expiry'
                )
    return rules


@dataclass(frozen=True)
class OverridePlan:
    """What the rules do to a selected suite: which rule governs each scenario × policy, which scenarios are not run, and the
    notes that need no run to be known."""

    rules: tuple[Rule, ...]
    today: date
    #: The scenarios the run selected and the policies of the eval set, in the order the caller gave them.
    scenarios: tuple[str, ...]
    policies: tuple[str, ...]
    #: Scenarios every scenario × policy of which a rule ignores: the caller does not run them.
    not_run: tuple[NotRun, ...]
    #: Expired rules, rules about to expire, rules that match nothing in the suite and fully ignored scenarios on the smoke list.
    notes: tuple[OverrideNote, ...]

    def rule_for(self, scenario: str, policy: str) -> Rule | None:
        """The rule that ignores the scenario × policy, or ``None``: the first active rule of the best rank."""
        return _winning(self.rules, self.today, scenario, policy)

    @property
    def run_scenarios(self) -> tuple[str, ...]:
        """The selected scenarios to run: those with at least one scenario × policy no rule ignores."""
        skipped = {n.scenario for n in self.not_run}
        return tuple(s for s in self.scenarios if s not in skipped)


def _winning(rules: Sequence[Rule], today: date, scenario: str, policy: str) -> Rule | None:
    live = [(rule.rank, index, rule) for index, rule in enumerate(rules) if rule.active_on(today) and rule.matches(scenario, policy)]
    return min(live, key=lambda entry: entry[:2])[2] if live else None


def _ignored(rule: Rule) -> Ignored:
    return Ignored(rule.describe(), rule.reason, rule.expires)


def plan_overrides(
    rules: Sequence[Rule],
    *,
    scenarios: Sequence[str],
    policies: Sequence[str],
    today: date,
    smoke: Sequence[str] = (),
    suite_scenarios: Sequence[str] | None = None,
) -> OverridePlan:
    """Resolve ``rules`` against the selected ``scenarios`` and the eval set's ``policies``.

    A scenario whose every policy is ignored is in :attr:`OverridePlan.not_run`, with every rule that ignores one of its policies.
    ``smoke`` names the scenarios on the smoke list; one of them that is not run is a note. ``suite_scenarios`` is every scenario
    of the committed set: a rule matches nothing only when it matches none of those, so a rule for a scenario outside this run's
    selection is not a dead rule. Without it the selection is the suite.
    """
    rules = tuple(rules)
    not_run: list[NotRun] = []
    for scenario in scenarios:
        winners = [_winning(rules, today, scenario, policy) for policy in policies]
        rules_of = [winner for winner in winners if winner is not None]
        if winners and len(rules_of) == len(winners):
            not_run.append(NotRun(scenario, tuple(policies), tuple(dict.fromkeys(_ignored(rule) for rule in rules_of))))
    notes: list[OverrideNote] = []
    suite = tuple(scenarios) if suite_scenarios is None else tuple(suite_scenarios)
    for rule in rules:
        if not rule.active_on(today):
            notes.append(OverrideNote('expired', rule.describe(), rule.expires))
        elif not any(rule.matches(s, p) for s in suite for p in policies):
            notes.append(OverrideNote('matches_nothing', rule.describe()))
        elif rule.expires is not None and (days_left := (rule.expires - today).days) <= EXPIRY_NOTE_DAYS:
            notes.append(OverrideNote('expires_soon', rule.describe(), rule.expires, days_left))
    notes += [OverrideNote('smoke_ignored', n.scenario) for n in not_run if n.scenario in smoke]
    return OverridePlan(rules, today, tuple(scenarios), tuple(policies), tuple(not_run), tuple(notes))


def _passes_every_trial(cell: Cell) -> bool:
    return cell.outcome.outcome == 'passed' and all(v in ('clear', 'not_applicable', 'deferred') for v in cell.verdicts)


def apply_overrides(run: Run, plan: OverridePlan) -> Run:
    """``run`` with each ignored scenario × policy marked, the not-run scenarios listed and the override notes added.

    A cell keeps its computed outcome; :func:`plurai.report.tally` and the renderers count an ignored cell toward nothing.
    A rule naming both a scenario and a policy whose every cell passes in every trial is noted as ready to remove.
    """
    cells = []
    governed: dict[Rule, list[Cell]] = {}
    for cell in run.cells:
        rule = plan.rule_for(cell.scenario, cell.policy)
        if rule is not None:
            governed.setdefault(rule, []).append(cell)
        cells.append(cell if rule is None else replace(cell, ignored=_ignored(rule)))
    passing = [
        OverrideNote('now_passing', rule.describe())
        for rule in plan.rules
        if rule.rank[0] == 0 and (held := governed.get(rule)) and all(_passes_every_trial(c) for c in held)
    ]
    return replace(run, cells=tuple(cells), not_run=plan.not_run, override_notes=(*plan.notes, *passing))
