"""One pass of the runner for ``plurai ci run``: the typed run API, called in this process, with the agent as its child.

A pass runs the named scenarios for ``trials`` trials into its own output directory and leaves the runner's artifacts there.
Nothing is written to the working directory: no run pointer, no sidecar. The environment of the agent child comes from the
SDK allowlist (:mod:`plurai.child_env`), so a GitHub credential in this process never reaches the agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from plurai._endpoints import Endpoint
from plurai.ci.gitrepo import head_commit
from plurai.gates.invocation import run_invocation

COMMAND = 'plurai ci run'


@dataclass(frozen=True)
class PassPlan:
    """What one pass runs."""

    simulate: Path
    scenario_set_id: str
    eval_set_id: str
    scenarios: Sequence[str]
    trials: int
    out: Path
    mock_service_url: str
    #: The handle of this pass: distinct per pass, because the runner's own default is a wall-clock second.
    run_name: str


def agent_version(workdir: Path) -> str:
    """The agent build tag the backend stamps on a thread: the short commit of the checkout, else ``unknown``."""
    return head_commit(workdir, short=True) or 'unknown'


def run_pass(plan: PassPlan, endpoint: Endpoint, *, version: str) -> int:
    """Run ``plan`` and return the runner's exit code: 0 or 1 when scenarios ran, anything else when the runner stopped."""
    # Imported here: the runner imports the ``plurai`` command line module that registers this package's commands.
    from plurai.run_simulation import RunConfig, run_suite, settings_for_run  # noqa: PLC0415

    config = RunConfig(
        scenario_set_id=plan.scenario_set_id,
        scenario_ids=list(plan.scenarios),
        agent_module_path=plan.simulate,
        out=plan.out,
        mock_service_url=plan.mock_service_url,
        eval_set_id=plan.eval_set_id,
        max_turns=None,
        concurrency=4,
        upload_datasets=False,
        run_name=plan.run_name,
        run_phase='full',
        agent_version=version,
        trials=plan.trials,
        stability_retry=False,
        replace=True,
        no_run_pointer=True,
    )
    settings = settings_for_run(endpoint, mock_service_url=plan.mock_service_url)
    return run_invocation(
        command=COMMAND,
        overrules_file=None,
        record=None,
        body=lambda: run_suite(config, settings, mock_pin_source='--mock-service-url'),
    )
