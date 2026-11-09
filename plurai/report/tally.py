"""The counts a heading shows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from plurai.report.outcome import CLASS_ORDER, NotMeasuredClass
from plurai.report.run import Credits, Run

#: More than this share of the selected gating scenarios quarantined as unsolvable leaves the run unmeasured: the suite,
#: not the agent, is what failed.
QUARANTINE_CAP_PERCENT = 20


@dataclass(frozen=True)
class Tally:
    failed: int
    unstable: int
    advisory: int
    #: Gating cells only: an advisory cell that could not be measured is listed, never counted.
    not_measured: int
    passed: int
    not_applicable: int
    #: The causes present among the counted not-measured cells, in :data:`CLASS_ORDER`.
    not_measured_classes: tuple[NotMeasuredClass, ...]
    #: The credit refusal behind the first counted credits cell, when the run carries one.
    credits: Credits | None = None
    #: Scenarios quarantined as unsolvable, and how many scenarios the run selected for gating: those with a gating
    #: scenario × policy plus every quarantined one, which has no tags to say otherwise.
    quarantined: int = 0
    gating_scenarios: int = 0
    #: How many scenario × policy were judged.
    cells: int = 0

    @property
    def quarantine_capped(self) -> bool:
        """Whether quarantined scenarios exceed :data:`QUARANTINE_CAP_PERCENT` of the selected gating scenarios."""
        return 100 * self.quarantined > QUARANTINE_CAP_PERCENT * self.gating_scenarios

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
    """Count the outcomes of ``run``'s cells."""
    counts = dict.fromkeys(('failed', 'unstable', 'advisory', 'passed', 'not_applicable'), 0)
    unmeasured = 0
    classes: set[str] = set()
    credits: Credits | None = None
    for cell in run.cells:
        result = cell.outcome
        if result.outcome != 'not_measured':
            counts[result.outcome] += 1
        elif result.level == 'gating':
            unmeasured += 1
            classes.update(result.classes)
            credits = credits or cell.credits
    gating = {c.scenario for c in run.cells if c.outcome.level == 'gating'}
    return Tally(
        not_measured=unmeasured,
        not_measured_classes=tuple(c for c in CLASS_ORDER if c in classes),
        credits=credits,
        quarantined=len(run.quarantined),
        cells=len(run.cells),
        gating_scenarios=len(gating) + len(run.quarantined),
        **counts,
    )
