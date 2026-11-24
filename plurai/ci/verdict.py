"""One mapping from a completed run, or from the stop that ended it, to an exit code and a heading.

``plurai ci run`` ends at the first stop it meets, in the order below, and writes its reports first. The one run with no
reports is a run with no ``PLURAI_CI_OUT_DIR``: it has nowhere to write them and exits 65.

=====  ==================================================================================  ==================
Order  Stop                                                                                Exit
=====  ==================================================================================  ==================
1      a bad flag value, ``ci.yaml``, an id file, the env alias, an ``agent.env`` name,    65 (refused)
       the SDK pin
2      no ``PLURAI_TOKEN``, or an empty name declared in ``agent.env``                     20 (no secrets)
3      a backend address or token the resolver refuses                                     65 (refused)
4      a committed id missing, unpublished or outside the lineage, a policy tier the SDK   20 (suite)
       does not know, an eval set with no policies, a scenario set with no scenarios
4      the org has no credit                                                               20 (credits)
4      a ``smoke`` name the scenario set does not have                                     65 (refused)
4      the mock answers 410, an edit session holds the env, or the mock does not answer    20 (suite, infra)
       in time
5      too many quarantined scenarios, or nothing selected to run                          20 (suite)
6      the runner leaves no summary                                                        20 (infra), or 10
6      a retry above the cap, or a retry pass without a summary, when nothing gating       20 (infra)
       failed in the first pass
=====  ==================================================================================  ==================

Three stops can come at any call, so they have no order: a 401 or a 403 (77), a 426 (10) and a backend that does not answer
after the retries (20, infra). A refusal that comes after the passes started keeps the results measured so far in ``run.json``.
A signal cancels the open threads and ends the run with the partial results (20), and an unexpected error in the SDK is
reported as not measured (20, infra).

A run that completes exits 1 when any gating scenario × policy failed, else 20 when any gating one is not measured, else 0.
A real failure shows as a failure even when other scenario × policy could not be measured, and one that a ``ci.yaml`` rule
ignores counts toward neither. A gating scenario × policy violated in some but not all trials is shown as unstable and does
not change the exit. One violated in every trial it measured, with the others lost to infra causes a second pass may clear,
is retried, and counts as failed when the retry cannot measure the lost trials. A policy tier the SDK does not know, a policy
of the eval set a scenario has no verdict for and an agent that raised make the scenario × policy not measured, never a silent
pass.

A retry that cannot happen never hides a failure: when a gating scenario × policy failed in the first pass, the run exits 1 on
it with the retry problem as a note, and only a run with no gating failure ends on the retry problem as a stop.
"""

from __future__ import annotations

from dataclasses import dataclass

from plurai import exit_codes
from plurai.report.outcome import NotMeasuredClass
from plurai.report.render import ReportOptions, Stopped, render_heading, stopped_heading
from plurai.report.run import CREDIT_CODES, Run, cause_parts
from plurai.report.tally import tally

_EXIT_OF_RUN = {'failed': exit_codes.POLICIES_FAILED, 'not_measured': exit_codes.NOT_MEASURED, 'passed': exit_codes.OK}


@dataclass(frozen=True)
class Verdict:
    """How a run ends: the process exit code and the heading line that opens every rendering of it."""

    exit_code: int
    heading: str


def verdict(subject: Run | Stopped, options: ReportOptions | None = None) -> Verdict:
    """The exit code and heading of a completed run (read with :func:`plurai.report.read_run`, after overrides) or of a stop.

    ``options`` carries what the heading may name beside the counts, such as main's run for the "new vs main" mark.
    """
    if isinstance(subject, Stopped):
        return Verdict(subject.exit_code, stopped_heading(subject))
    if subject.cancelled:
        # A cancelled run measured only part of the suite, whatever the measured part says.
        return Verdict(exit_codes.NOT_MEASURED, render_heading(subject, options))
    return Verdict(_EXIT_OF_RUN[tally(subject).verdict], render_heading(subject, options))


def not_measured_class(cause: str, *, code: str | None = None) -> NotMeasuredClass:
    """The class a cause earns. A cause the SDK does not know is infra. ``credits`` is credits only for one of :data:`CREDIT_CODES`;
    any other 402 is infra. A cause joined with ``+`` takes the class of its worst part (infra, else credits, else agent, else
    suite)."""
    if cause == 'credits':
        return 'credits' if code in CREDIT_CODES else 'infra'
    classes = {part.cls for part in cause_parts(cause)}
    return 'infra' if 'infra' in classes else 'credits' if 'credits' in classes else 'agent' if 'agent' in classes else 'suite'
