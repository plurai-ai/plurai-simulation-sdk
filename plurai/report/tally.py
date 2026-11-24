"""The counts a heading shows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from plurai.report.outcome import CLASS_ORDER, NotMeasuredClass
from plurai.report.run import Credits, Run

#: More than this share of the selected gating scenarios quarantined as unsolvable leaves the run unmeasured: the suite,
#: not the agent, is what failed.
QUARANTINE_CAP_PERCENT = 20


def over_quarantine_cap(quarantined: int, gating: int) -> bool:
    """Whether ``quarantined`` scenarios are more than :data:`QUARANTINE_CAP_PERCENT` of ``gating`` selected gating scenarios."""
    return 100 * quarantined > QUARANTINE_CAP_PERCENT * gating


def newly_quarantined(run: Run, main: Run | None) -> tuple[str, ...]:
    """The scenarios ``run`` quarantined that main's last run did not. Without a main run to compare, none: it is a display."""
    return () if main is None else tuple(name for name in run.quarantined if name not in main.quarantined)


@dataclass(frozen=True)
class Tally:
    #: The gating cells that failed: these are what fails the check.
    failed: int
    #: The gating cells that were unstable. They are shown and never change the verdict.
    unstable: int
    #: Gating cells only: an advisory cell that could not be measured is listed, never counted.
    not_measured: int
    passed: int
    not_applicable: int
    #: The advisory cells that failed or were unstable. They never change the verdict.
    advisory_failed: int
    advisory_unstable: int
    #: Cells every valid trial of which the backend deferred to a sibling policy: counted beside N/A, in no other count.
    deferred: int
    #: The causes present among the counted not-measured cells, in :data:`CLASS_ORDER`.
    not_measured_classes: tuple[NotMeasuredClass, ...]
    #: The credit refusal behind the first counted credits cell, when the run carries one.
    credits: Credits | None = None
    #: Scenarios quarantined as unsolvable, and how many scenarios the run selected for gating: those with a gating
    #: scenario × policy plus every quarantined one, which has no tags to say otherwise.
    quarantined: int = 0
    gating_scenarios: int = 0
    #: How many scenario × policy were judged and count: those a ``ci.yaml`` rule ignores are in :attr:`ignored` instead.
    cells: int = 0
    #: How many scenario × policy a rule ignores. They count toward nothing above.
    ignored: int = 0
    #: How many scenarios no rule leaves a scenario × policy to run in, so they were not run.
    not_run: int = 0

    @property
    def quarantine_capped(self) -> bool:
        """Whether quarantined scenarios exceed :data:`QUARANTINE_CAP_PERCENT` of the selected gating scenarios."""
        return over_quarantine_cap(self.quarantined, self.gating_scenarios)

    @property
    def classes(self) -> tuple[NotMeasuredClass, ...]:
        """The causes behind a not-measured run, in :data:`CLASS_ORDER`: those of the counted cells, and suite when capped
        or when nothing was judged."""
        present = set(self.not_measured_classes) | ({'suite'} if self.quarantine_capped or self.cells == 0 else set())
        return tuple(c for c in CLASS_ORDER if c in present)

    @property
    def verdict(self) -> Literal['failed', 'not_measured', 'passed']:
        """What the run concluded, the one place the precedence lives: a failed gating scenario × policy is a finding and wins
        even when the run is also not measured; else nothing judged, a quarantine cap or an unmeasured gating scenario × policy
        leaves it not measured."""
        if self.failed:
            return 'failed'
        if self.not_measured or self.quarantine_capped or self.cells == 0:
            return 'not_measured'
        return 'passed'


def tally(run: Run) -> Tally:
    """Count the outcomes of ``run``'s cells: failed and unstable by level, the rest by outcome. A scenario × policy a rule
    ignores is counted in :attr:`Tally.ignored` and nowhere else."""
    counts = dict.fromkeys(('failed', 'unstable', 'advisory_failed', 'advisory_unstable', 'passed', 'not_applicable', 'deferred'), 0)
    unmeasured = 0
    classes: set[str] = set()
    credits: Credits | None = None
    counted = [cell for cell in run.cells if cell.ignored is None]
    for cell in counted:
        result = cell.outcome
        if result.outcome == 'not_measured':
            if result.level == 'gating':
                unmeasured += 1
                classes.update(result.classes)
                credits = credits or cell.credits
        elif result.level == 'advisory' and result.outcome in ('failed', 'unstable'):
            counts[f'advisory_{result.outcome}'] += 1
        else:
            counts[result.outcome] += 1
    gating = {c.scenario for c in counted if c.outcome.level == 'gating'}
    return Tally(
        not_measured=unmeasured,
        not_measured_classes=tuple(c for c in CLASS_ORDER if c in classes),
        credits=credits,
        quarantined=len(run.quarantined),
        cells=len(counted),
        ignored=len(run.cells) - len(counted),
        not_run=len(run.not_run),
        gating_scenarios=len(gating) + len(run.quarantined),
        **counts,
    )
