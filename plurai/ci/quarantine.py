"""Scenarios quarantined as unsolvable: not run, not scenario × policy, a note on every report.

A quarantined scenario never changes the check by itself. When more than :data:`QUARANTINE_CAP_PERCENT` of the gating scenarios
are quarantined, the run stops as not measured (suite): a broken check could otherwise shrink the gate to a green remainder. The
gating scenarios are counted as the report counts them: those that run with a scenario × policy no ``ci.yaml`` rule ignores and
the fail-on tags make gating, plus every quarantined scenario. The same count before and after the run keeps a run that passed
the cap from ending capped after it spent the credits.
"""

from __future__ import annotations

from collections.abc import Sequence

from plurai.report.render import Stopped
from plurai.report.tally import QUARANTINE_CAP_PERCENT, newly_quarantined, over_quarantine_cap

__all__ = ['QUARANTINE_CAP_PERCENT', 'newly_quarantined', 'quarantine_stop']


def quarantine_stop(selected_gating: Sequence[str], quarantined: Sequence[str]) -> Stopped | None:
    """The stop for too many quarantined scenarios, or ``None`` when the run may go on.

    ``selected_gating`` names the scenarios to run that have a gating scenario × policy; ``quarantined`` those the artifact
    read stamps unsolvable, each of which counts as gating whether or not it is listed. The cap is strict: exactly
    :data:`QUARANTINE_CAP_PERCENT` percent goes on.
    """
    gating = set(selected_gating) | set(quarantined)
    flagged = len(set(quarantined))
    return Stopped.too_many_quarantined(flagged, len(gating)) if over_quarantine_cap(flagged, len(gating)) else None
