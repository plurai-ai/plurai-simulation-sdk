"""``plurai.ci``: the CI simulation commands (``plurai ci run``, ``ci validate``, ``ci token`` and ``suite diff``), and the
verdict layer they share, exported here as pure functions over a ``run.json``.

The verdict layer reads no network, clock or process: the caller passes today's date and the documents.

* :mod:`plurai.ci.overrides` parses the ``overrides`` of ``ci.yaml`` and applies them to a run.
* :mod:`plurai.ci.retry` picks the scenarios worth one more pass for an infra cause and merges that pass in.
* :mod:`plurai.ci.quarantine` decides when quarantined scenarios leave the run unmeasured.
* :mod:`plurai.ci.verdict` maps a completed run or a stop to an exit code and a heading.

The other modules touch the world: :mod:`~plurai.ci.config` and :mod:`~plurai.ci.gitrepo` read the project and git,
:mod:`~plurai.ci.backend`, :mod:`~plurai.ci.online`, :mod:`~plurai.ci.mock_wait` and :mod:`~plurai.ci.cancel` call the backend,
:mod:`~plurai.ci.runner` starts the agent, :mod:`~plurai.ci.interrupt` handles signals and :mod:`~plurai.ci.token` calls the
login's key routes and ``gh``. :mod:`~plurai.ci.run` sequences them and :mod:`~plurai.ci.cli` registers the commands.
"""

from plurai.ci.overrides import OverridePlan, OverridesError, Rule, apply_overrides, parse_overrides, plan_overrides
from plurai.ci.quarantine import newly_quarantined, quarantine_stop
from plurai.ci.retry import RETRY_CAP_PERCENT, RetrySelection, merge_retry, retry_stop, select_retries
from plurai.ci.verdict import Verdict, not_measured_class, verdict

__all__ = [
    'RETRY_CAP_PERCENT',
    'OverridePlan',
    'OverridesError',
    'RetrySelection',
    'Rule',
    'Verdict',
    'apply_overrides',
    'merge_retry',
    'newly_quarantined',
    'not_measured_class',
    'parse_overrides',
    'plan_overrides',
    'quarantine_stop',
    'retry_stop',
    'select_retries',
    'verdict',
]
