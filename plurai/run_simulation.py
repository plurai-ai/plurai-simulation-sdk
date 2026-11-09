"""``plurai-run-simulation`` — client-side end-to-end runner.

Runs the customer's ``simulate.py`` against the mock service (via the SDK child harness)
and drives the SDK-free Plurai backend orchestration server for the LLM user-turns and
policy eval, then writes the per-scenario artifacts the diagnose flow reads. Everything
runs CLIENT-side, so the orchestration server may be local or remote.

Scenarios and eval-sets are referenced **by ID**; the orchestration server resolves them
from its shared backend storage (what the backend's scenario and eval generation wrote).
The client supplies only the agent (``simulate.py``) — it needs no local copy of the scenarios.

Per scenario ``<out>/<scenario_id>/``:
  - transcript.json        user + (rich) agent messages, turn_count, policy results
  - final_world_state.json the server-normalized {entities: {...}} snapshot
  - tool_events.json        the mock's recorded tool-event log, UNIONED with the calls
                            recovered from the agent's own message chains. Only
                            ``mode: mock`` tools dispatch through ``/mock/execute`` and
                            leave a first-hand event; a ``mode: real`` tool (and every
                            tool under ``mockless``) is observable ONLY via that
                            reconstruction, so its records carry ``_recovered: true``
                            and an agent that narrates no tool call leaves no trace of
                            one at all. "Did tool X fire?" must never be asked of the
                            mock lane alone — see
                            ``plurai.observability.tool_call_recovery``.
  - error.json              written instead, on a crashed run (the diagnose flow's
                            highest-precedence failure signal)
  - error_dialog.json       beside error.json when the failed attempt reached a turn: the
                            turns it reached (``ERROR_DIALOG_FILENAME``; its count rides
                            error.json as ``turns_reached``)
  - transcript.partial.json, tool_events.partial.json
                            beside error.json: the conversation and tool events the
                            failed attempt had produced (``PARTIAL_EVIDENCE_FILES``)

The backend and its token come from ``plurai login`` (or ``PLURAI_URL`` / ``PLURAI_TOKEN``); there is no flag for them.

Usage::

    plurai-run-simulation \\
        --scenario-set-id <uuid> --scenario-id scn_01 --scenario-id scn_02 \\
        --agent-module-path ./plurai_sim/simulate.py \\
        --out ./runs \\
        --mock-service-url <backend url>/mock/<env_id>/ \\
        --eval-set-id <uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Container, Coroutine, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from dataclasses import replace as dataclasses_replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, Protocol, cast, get_args
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from plurai import _http, client_handshake, command_run, exit_codes
from plurai import _subprocess as _child_process
from plurai._endpoints import (
    CREDENTIAL_FIX,
    TOKEN_ENV,
    URL_ENV,
    Endpoint,
    EndpointError,
    resolve as resolve_endpoint,
)
from plurai._dotenv import resolve_dotenv_path
from plurai._platform import utf8_child_text, utf8_stdio
from plurai._mocking.client import DEFAULT_MOCK_SERVICE_URL
from plurai._mocking.codecs import SUPPRESSED_KIND, _plain
from plurai._mocking.turn_context import MockRouteCause
from plurai._mocking.resolution import (
    ABORTING_ROUTE_CAUSES,
    UnansweredCall,
    abort_remedy,
    aborting_route_cause,
    capacity_block_of_call,
    unavailable_block_of_call,
    credit_block_of_call,
    unanswered_call_remedy,
)
from plurai._protocol import (
    TRAIL_NOT_DISPATCHABLE_STATUSES,
    HarnessIpcOverrun,
    MockMode,
    MockRuntimeFaultRaised,
    MockStubFallback,
    ScenarioQuarantined,
    is_harness_owned,
    restart_shaped_calls,
)
from plurai.agent_io import DEFAULT_AGENT_IO, AgentIOConfigError, declared_agent_io
from plurai.egress import (
    ProbeMode,
    enforcement_layer,
    load_declaration,
    run_egress_preflight,
)
from plurai.egress_guard import GUARD_ENV
from plurai.customer_safe import describe_body, describe_failure, error_id_of, response_json
from plurai.gates.apply import make_finding
from plurai.gates.catalogue import (
    CONFIG_UNREADABLE,
    DIALOG_TURN_FAILURE_LANDED,
    DIALOG_WORLD_CAPTURE_FAILED,
    EGRESS_GUARD_BLOCKED,
    EGRESS_GUARD_COVERAGE_GAP,
    EGRESS_GUARD_UNCONFIRMED,
    EGRESS_PREFLIGHT_CRASHED,
    EGRESS_UNGUARDED_UNMEASURED,
    HARNESS_AGENT_IO_MALFORMED,
    HARNESS_PIPELINE_UNDECLARED,
    HARNESS_SHAPE_OVERRIDE_CONTRADICTS_DECLARATION,
    HOST_REQUIREMENT_GAP,
    IPC_LINE_OVERRUN,
    IPC_REPLY_TRIMMED,
    MOCK_INTERCEPTION_CHANGED,
    MOCK_TOOL_CALL_UNANSWERED,
    PREFLIGHT_BACKEND_AUTH_FAILED,
    PREFLIGHT_BACKEND_REDIRECTED,
    PREFLIGHT_BACKEND_UNREACHABLE,
    PREFLIGHT_BACKEND_REJECTS_CREDENTIALS,
    PREFLIGHT_BACKEND_UNHEALTHY,
    PREFLIGHT_MOCK_REJECTS_CREDENTIALS,
    PREFLIGHT_MOCK_UNREACHABLE,
    PREFLIGHT_MOCK_URL_UNUSABLE,
    RUN_OUT_OCCUPIED,
    RUN_SCAN_DRIFT_UNCHECKED,
    RUN_SCAN_SOURCE_DRIFT,
    WORLD_USER_SURFACES_MALFORMED,
    WORLD_USER_SURFACE_ENTRY_UNREADABLE,
    WORLD_VALUES_COERCED,
)
from plurai.gates.cli import RUNNER_COMMAND, OverruleFileError, add_overrule_argument, load_overrules
from plurai.gates.contract import HARD_AUTHORITIES, GateFinding
from plurai.gates.invocation import GATE_AUTHORITY_FILENAME, RecordTarget, current as current_gate_authority, held_exit, run_invocation
from plurai.host_effects import (
    HOST_EFFECTS_KEY,
    READ_CHANNEL_TOOLS,
    READ_CONFINED,
    READ_LANDINGS,
    WRITE_LANDINGS,
    cli_own_write_landed,
    declared_host_requirements,
    host_read_scope,
    observed_writes,
    redact_cli_credentials,
    redact_value,
    redacted_process_output,
    register_backend_token,
    scrub_parent_record,
    trail_recorded,
)
from plurai.host_toolchain import PROBE_TIMEOUT_SECONDS as HOST_PROBE_TIMEOUT_SECONDS
from plurai.host_toolchain import host_findings, host_stamp, install_block
from plurai.identity import sdk_identity
from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, discover_config_documents
from plurai.user_surfaces import UserSurfacesConfigError, discover_surface_types
from plurai.observability import create_observability_sink
from plurai.observability.dataset_upload import DATASET_UPLOAD_METADATA_FILENAME, UploadResult, load_upload_metadata
from plurai.observability.rollup import _zero_result
from plurai.observability.scenario_verdict import read_scenario_verdict
from plurai.observability.sink_protocol import DatasetUrlSink, ExperimentLifecycleSink, ExperimentRunnerSink, ObservabilitySink, RunLinks, RunLinksSink
from plurai.observability.tool_call_recovery import TOOL_DECISION_EVENT
from plurai.orchestrator import (
    ADAPTER_DEGRADED_REASON,
    AGENT_ENDED_REASON,
    AGENT_RAISED_REASON,
    AGENT_RECURSION_REASON,
    GIVE_UP_CAUSE_AGENT_CANNOT,
    GIVE_UP_CAUSE_AGENT_DIVERGED,
    GIVE_UP_CAUSE_AGENT_REFUSED,
    GIVE_UP_CAUSE_AGENT_REPEATED,
    GIVE_UP_CAUSE_REPLY_NOT_VISIBLE,
    HANDED_OFF_REASON,
    HARNESS_CUT_REASONS,
    PLAN_EXHAUSTED_REASON,
    POLICY_TERMINATED_REASON,
    PROVIDER_INFRA_REASON,
    REPLAY_CUT_REASON,
    SIM_DEGENERATE_REASON,
    TIME_BUDGET_REASON,
    TURN_BUDGET_REASON,
    TURN_CAP_REASON,
    USER_GAVE_UP_REASON,
    Orchestrator,
    OrchestratorAdapterError,
    ScenarioTimeout,
    ThreadResult,
)
from plurai._http_retry import backoff_seconds, is_name_not_found
from plurai.capacity import (
    BACKEND_REFUSED_PHRASE,
    CAPACITY_TEXT,
    UNAVAILABLE_TEXT,
    CapacityBlock,
    UnavailableBlock,
    UnavailableCause,
    capacity_block_of,
    unavailable_block_of,
)
from plurai.credit import CREDIT_REFUSED_STATUS, CreditBlock, CreditKind, credit_block_from_body, credit_block_of, credit_sentence
from plurai.login._environments import billing_url_for
from plurai.orchestrator._retry import (
    NON_OUTAGE_TIMEOUTS,
    TRANSIENT_STATUS_CODES,
    CreditRefused,
    NetworkRefused,
    RetryConfig,
    ServerHTTPError,
    is_transient_for_idempotent,
    job_refused_for_credit,
)
from plurai.orchestrator.client import build_default_client, eval_set_policy_names, submit_carrying_overrules, take_gate_authority
from plurai.orchestrator.models import EvalSetPoliciesResponse, FinalizeResponse

# The ONE reader of the child's finalize-error marker. A second derivation here is how two
# consumers come to disagree about whether a snapshot materialized.
from plurai.orchestrator.orchestrator import (
    DIALOG_SHAPE,
    PIPELINE_SHAPE,
    SHARED_WORLD_RISK_KIND,
    WORLD_OWNER_AGENT,
    WORLD_OWNER_EXTERNAL,
    WORLD_OWNER_MOCK,
    WORLD_SNAPSHOT_HOOK,
    ScenarioProgress,
    Shape,
    _declared_native_tools,
    _declared_single_instance,
    _declared_world_owner,
    _declared_world_slots,
    _finalize_error_message,
    _module_scope_bindings_of_file,
    contradicted_shape_declaration,
    reset_warning_ledgers,
    shared_world_risk,
    turn_input_findings,
    undeclared_pipeline_refusal,
    unobservable_world_warning,
    unreset_external_world_finding,
)
from plurai.orchestrator.redrive import (
    REDRIVE_BUDGET,
    RedriveProvenance,
    RedriveTrigger,
    backend_outage,
    classify_restart_failure,
    seed_fault,
    wait_until_ready,
)
from plurai.world.readback import SEED_NOT_LANDED_REASON
from plurai.run_artifacts import (
    ERROR_DIALOG_FILENAME,
    ERROR_RECORD_FILES,
    LC_MESSAGE_TYPES,
    PARTIAL_EVIDENCE_FILES,
    PARTIAL_TOOL_EVENTS_FILE,
    PARTIAL_TRANSCRIPT_FILE,
    PREVIOUS_ATTEMPT_DIRNAME,
    RUN_SUMMARY_FILENAME,
    SCENARIO_ARTIFACT_FILES,
    STABILITY_RETRY_DIRNAME,
    TRANSCRIPT_SCHEMA,
    TRIAL_DIR_PREFIX,
)
from plurai.scan_contract_cli import BadUsageError, admitted_suffixes, read_recorded_inventory, recheck_source_drift
from plurai.settings import (
    FULL_RUN_SCENARIO_TIME_BUDGET,
    MIN_SECONDS,
    PREFLIGHT_ATTEMPTS,
    PREFLIGHT_CONNECT_TIMEOUT,
    PREFLIGHT_RETRY_DELAY,
    VENDOR_TIMEOUT_HEADROOM_SECONDS,
    NetworkSettings,
    ObservabilityPosture,
    RunPhase,
    Settings,
    SinkName,
    default_agent_turn_timeout,
    default_scenario_time_budget,
    derive_scenario_hard_ceiling,
)
from plurai.trials import (
    TRIALS_SCHEMA,
    ScenarioCells,
    aggregate_policy_turn_rates,
    aggregate_regression_replays,
    aggregate_trials,
    cells_from_run_dir,
    cells_from_summary,
    harness_cut_finding,
    is_regression_scenario,
    repeated_losses,
    run_totals,
    trial_dir_name,
    trial_run_name,
)
from plurai.user_path import expand_user_path
from plurai import scoring, verdicts

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` ATOMICALLY — write a sibling temp, then ``replace``.

    ``replace`` is a rename within the same directory, so a reader either sees the whole
    previous file or the whole new one, never a truncated prefix. This matters most for
    ``run_summary.json``, which is the SOLE record of a previous invocation's rows
    (``_merge_prior_scenarios`` reads it back): a plain ``write_text`` torn by a crash or a
    full disk mid-retry destroys that population permanently — the next merge hits the
    unreadable-file branch and degrades to this-run-only, which is the shrinking total the
    merge exists to prevent. The same temp-then-``replace`` convention the anonymization
    egress path uses (``ingest/anonymize.py``, ``ingest/cli.py``), for the same reason.
    """
    tmp = path.with_name(f'{path.name}.tmp')
    tmp.write_text(redact_cli_credentials(json.dumps(payload, indent=2, default=str)), encoding='utf-8')  # an artifact is uploaded: no credential rides it
    tmp.replace(path)  # atomic in-place swap


#: The two statuses ``GET /eval-sets/{id}/policies`` documents, and the only two the
#: pre-flight treats as fatal. See :func:`_preflight_eval_set` for the full posture.
_EVAL_SET_FATAL_STATUSES = frozenset({400, 404})

#: The eval-set state that means the bytes about to score this run are an open edit session's
#: mid-edit working copy — not the committed set. The one value the pre-flight warns on.
_EVAL_SET_DRAFT = 'draft'

#: The state a backend reports when it has no edit-session store wired, so it cannot tell a
#: committed set from a draft. Not a warning about the run; a NOTE that the axis is dark.
_EVAL_SET_UNKNOWN = 'unknown'

#: The states the backend documents on that route. Held only so a value from OUTSIDE it can
#: be REPORTED as one — a closed set that silently ignores what falls outside it is how a new
#: server state would read as "not a draft, carry on". Never used to reject a response.
_EVAL_SET_STATES = frozenset({_EVAL_SET_DRAFT, 'promoted', _EVAL_SET_UNKNOWN})

#: WHICH eval-set bytes this run was launched against, for ``run_summary.json``. Same
#: module-level handoff as ``_LAST_EGRESS_RECORD`` and for the same reason: the pre-flight runs
#: before the strategies and the summary is written after them, with no frame spanning both.
#: Written to the summary ALWAYS, empty included — "the committed set", "a mid-edit draft" and
#: "nobody asked" are three different runs, and an absent key makes the third read like the first.
_LAST_EVAL_SET_RECORD: dict[str, Any] = {}


def _server_detail(response: Any) -> str:
    """The server's own explanation of a rejection, for the operator-facing line: its scrubbed message and error id.

    Never the raw body: a body that carries no message contributes nothing. Nothing here may raise on the
    failure path it exists to describe.
    """
    return describe_failure(response, limit=500, empty='')


#: Where the egress preflight's own record lands, beside ``run_summary.json``. Its own file
#: rather than a key on the summary: the summary is a verdict about the RUN, and this is a
#: statement about the machine the run happened on — it is written even when the run is refused
#: and no verdict exists.
EGRESS_PREFLIGHT_FILENAME = 'egress_preflight.json'

#: What the Python agent child's in-process guard saw, one JSON row per record (``plurai.egress_guard``),
#: folded into ``run_summary.egress.runtime`` by :func:`_egress_summary`.
EGRESS_OBSERVED_FILENAME = 'egress_observed.jsonl'


#: The last egress verdict this process computed, so ``run_summary.json`` can carry it. A
#: module-level handoff rather than a threaded parameter because the preflight runs before the
#: strategies and the summary is written after them, with no frame spanning both — the same
#: shape ``_mocking/resolution`` uses for its per-session counters. Consumers read
#: ``run_summary.json``; a file nothing in this repo reads is not a channel.
_LAST_EGRESS_RECORD: dict[str, Any] = {}
#: The declared production host NAMES of that verdict, beside it rather than in it: the summary
#: carries counts, and :func:`_egress_summary` needs the names to tell an observed host the
#: declaration never named (``undeclared_observed``) from one it did.
_LAST_EGRESS_DECLARED: set[str] = set()


def _preflight_egress(cfg: _RunConfig, settings: Settings, *, keep_guard_log: bool) -> None:
    """Arm the egress guard from the declaration, then measure whether the agent can reach PRODUCTION.

    ENFORCEMENT FIRST, AND FROM THE DECLARATION ALONE. The in-process guard (:mod:`plurai.egress_guard`)
    is the isolation itself, so it is armed before the source scan and the probe run, from a deny list
    computed only from the declaration (declared hosts − ``allow_hosts`` − this run's own infrastructure
    − loopback), and nothing below can disarm it: not a crashed probe, not a partly malformed block, not
    an overrule. A malformed declaration fails closed for enforcement (every host that parses is
    denied) and is only advisory for the verdict.

    THE MEASUREMENT IS A FINDING, NEVER A REFUSAL. Every verdict — including a crashed preflight — is
    printed, written to ``egress_preflight.json`` and reported to the invocation's gate collector
    (:meth:`plurai.egress.EgressPreflight.findings`). The caller runs the one gate funnel right after:
    the egress findings that hold the run, until an ``--overrule`` bound to the fingerprint lifts them, are a
    reachable declared host with NO enforcement layer (``egress.breached_unenforced``), a declaration that
    leaves the guard's deny list missing declared hosts (``egress.deny_list_incomplete``), and a crash that
    leaves the agent neither guarded nor measured (``egress.unguarded_unmeasured``).

    The allow list this frame contributes is the run's OWN infrastructure: the mock service and the
    Plurai backend are hosts the agent is SUPPOSED to reach, and only this frame knows which they are.
    """
    gates = current_gate_authority()
    os.environ.pop(GUARD_ENV, None)
    _child_process.EGRESS_GUARD_UNCONFIRMED = None
    allowed: list[str] = []
    for url in (cfg.mock_service_url, settings.endpoint.url):
        if url and (host := urlsplit(url).hostname):
            allowed.append(host)
    mock_mode = str(getattr(settings.network, 'mock_mode', None) or 'connect')
    harness_suffix = Path(cfg.agent_module_path).suffix
    try:
        declaration = load_declaration(cfg.agent_module_path)
        enforcement = enforcement_layer(cfg.agent_module_path, declaration, mock_mode=mock_mode)
    except Exception as exc:  # noqa: BLE001 — nothing to arm from; recorded as a crashed preflight
        _egress_crashed(cfg, gates, exc, phase='declaration', enforcement=None)
        gates.emit(_unguarded_unmeasured(exc, phase='declaration'))
        return
    deny, env = declaration.deny_hosts(allowed), [var for var, _ in declaration.env_hosts]
    _arm_child_guard(
        cfg.out,
        deny=deny,
        env=env,
        allowed=declaration.guard_allow(allowed),
        blocking=enforcement == 'python_guard',
        python=harness_suffix == '.py',
        keep=keep_guard_log,
    )
    if enforcement == 'python_guard' and (deny or env):
        # The guard is a claim about the CHILD, so each child confirms it in its Init reply — keyed on the
        # guard being armed with something to deny, never on the measurement below (which may crash).
        _child_process.EGRESS_GUARD_UNCONFIRMED = _guard_withdrawal(cfg, gates, tuple(allowed), harness_suffix=harness_suffix, mock_mode=mock_mode)
    try:
        preflight = run_egress_preflight(cfg.agent_module_path, mode=cfg.egress_probe, enforcement=enforcement, extra_allowed=tuple(allowed))
    except Exception as exc:  # noqa: BLE001 — a crashed measurement is a finding; it holds only where no enforcement layer stands in
        _egress_crashed(cfg, gates, exc, phase='probe', enforcement=enforcement)
        for finding in declaration.holding_findings():
            gates.emit(finding)
        if enforcement == 'none' and (declaration.hosts or declaration.env_hosts):
            # Production hosts are declared, nothing enforces isolation for this agent (TS / subprocess / mockless / a
            # non-`tools` skill channel), and the probe that would have measured them raised: the agent would run neither
            # guarded nor measured. An empty declaration has nothing to guard or measure, so it holds nothing.
            gates.emit(_unguarded_unmeasured(exc, phase='probe'))
        return
    _record_egress(cfg.out, preflight.as_record(), repo=_bench_repo(cfg))
    for finding in preflight.findings(harness_suffix=harness_suffix, mock_mode=mock_mode):
        gates.emit(finding)
    for line in preflight.report_lines():
        print(line, file=sys.stderr)


def _egress_crashed(cfg: _RunConfig, gates: Any, exc: Exception, *, phase: str, enforcement: str | None) -> None:
    """A preflight that raised: recorded as ``unprovable`` and reported as a finding. Never a refusal.

    The evidence is the exception CLASS and the phase only; the message moves between runs, so it rides
    ``observed`` and the record, never the fingerprint. ``enforcement`` rides the record when it was
    resolved, so the run-time coverage check still knows the guard was armed.
    """
    detail = f'the preflight raised: {type(exc).__name__}: {exc}'
    record = {'status': 'unprovable', 'probe_mode': cfg.egress_probe, 'detail': detail, **({'enforcement': enforcement} if enforcement else {})}
    _record_egress(cfg.out, record, repo=_bench_repo(cfg))
    print(f'[egress] WARNING: the egress preflight could not run ({type(exc).__name__}: {exc}); isolation is not established.', file=sys.stderr)
    gates.emit(
        make_finding(
            EGRESS_PREFLIGHT_CRASHED,
            subject='egress',
            evidence={'exc_type': type(exc).__name__, 'phase': phase},
            observed={'detail': detail},
            remedy='The egress preflight raised, so this run carries no isolation measurement. Fix what the detail names (usually the '
            '`egress` block in plurai.config.yaml) and re-run.',
            draft_ref=None,
        )
    )


def _unguarded_unmeasured(exc: Exception, *, phase: str) -> GateFinding:
    """The HOLDING finding for a crash that leaves nothing in place of isolation: no guard armed from the
    declaration (``phase='declaration'``), a probe that crashed for an agent no enforcement layer covers
    (``phase='probe'``), or a child that never confirmed the guard whose re-probe crashed
    (``phase='guard_withdrawal'``). Evidence is the class and the phase only; the message rides ``observed``."""
    return make_finding(
        EGRESS_UNGUARDED_UNMEASURED,
        subject='egress',
        evidence={'phase': phase, 'exc_type': type(exc).__name__},
        observed={'detail': f'{type(exc).__name__}: {exc}'},
        remedy='The egress preflight raised and nothing guards or measures this agent, so the run is held. Fix what the detail '
        'names (usually the `egress` block in plurai.config.yaml, or plurai in the agent interpreter) and re-run.',
        draft_ref=None,
    )


def _guard_withdrawal(cfg: _RunConfig, gates: Any, allowed: tuple[str, ...], *, harness_suffix: str, mock_mode: str) -> Callable[[], str | None]:
    """What a child that does NOT confirm the guard gets: `guarded` withdrawn, and the probe measures, once.

    ``gates`` is the invocation's collector, captured here because the hook runs on a worker thread. The
    re-probe runs with NO enforcement layer (the guard is what was withdrawn), so a reachable declared
    host is ``egress.breached_unenforced`` and holds this child under the normal rule: the returned line
    carries the finding's gate id and fingerprint, so an overrule can lift it on the re-run. A probe that
    crashes leaves the child neither guarded nor measured, so it holds the same way
    (``egress.unguarded_unmeasured``). Every finding is recorded once, whichever child asked first.
    """
    lock = threading.Lock()
    decided: list[str | None] = []
    why = (
        'an agent process did NOT confirm the in-process egress guard (an older plurai in the agent interpreter — '
        '`PLURAI_CHILD_PYTHON`?), so `guarded` is withdrawn and the probe decides; upgrade plurai there to >= 1.0.0'
    )

    def _withdraw() -> str | None:
        with lock:
            if not decided:
                gates.emit(
                    make_finding(
                        EGRESS_GUARD_UNCONFIRMED,
                        subject='egress',
                        evidence={'child_python': bool(os.environ.get('PLURAI_CHILD_PYTHON')), 'required_plurai': '>=1.0.0'},
                        observed={},
                        remedy='Upgrade plurai in the agent interpreter to >= 1.0.0 so the child arms the in-process guard.',
                        draft_ref=None,
                    )
                )
                findings: Sequence[GateFinding]
                try:
                    preflight = run_egress_preflight(cfg.agent_module_path, mode=cfg.egress_probe, enforcement='none', extra_allowed=allowed)
                except Exception as exc:  # noqa: BLE001 — no guard AND no probe: held, like a breach, until overruled
                    _egress_crashed(cfg, gates, exc, phase='guard_withdrawal', enforcement='none')
                    findings = (_unguarded_unmeasured(exc, phase='guard_withdrawal'),)
                else:
                    _record_egress(cfg.out, {**preflight.as_record(), 'warnings': [*preflight.warnings, why]}, repo=_bench_repo(cfg))
                    print(f'[egress] WARNING: {why}.', file=sys.stderr)
                    for line in preflight.report_lines():
                        print(line, file=sys.stderr)
                    findings = preflight.findings(harness_suffix=harness_suffix, mock_mode=mock_mode)
                held: list[str] = []
                seen = {(f.gate_id, f.fingerprint) for f in gates.findings}
                for finding in findings:
                    if (finding.gate_id, finding.fingerprint) not in seen:
                        gates.emit(finding)
                    if finding in gates.held():
                        held.append(f'held by gate {finding.gate_id} on {finding.subject!r}: {finding.remedy} [fingerprint {finding.fingerprint}]')
                decided.append(f'EGRESS PREFLIGHT HELD — {why}. ' + ' '.join(held) if held else None)
            return decided[0]

    return _withdraw


def _arm_child_guard(out: Path, *, deny: Sequence[str], env: Sequence[str], allowed: Sequence[str], blocking: bool, python: bool, keep: bool) -> None:
    """Ask every Python agent child this run spawns to arm ``plurai.egress_guard`` (inherited env).

    ``deny`` / ``env`` come from the declaration alone (:meth:`plurai.egress.Declaration.deny_hosts`), never
    from a probe, a flag or an overrule. ``blocking`` False (no Python guard applies to this agent): it only records.
    ``keep`` (a ``--scenario-id`` re-run into ``out``): the guard log keeps what earlier invocations' processes logged,
    so the run's egress block still covers the scenarios it carries; a full run starts it empty.
    """
    log = out / EGRESS_OBSERVED_FILENAME
    try:
        out.mkdir(parents=True, exist_ok=True)
        if keep:
            log.touch()
        else:
            log.write_text('', encoding='utf-8')
    except OSError as exc:
        print(f'[egress] WARNING: could not open {log} ({exc}); the in-process egress guard is NOT armed for this run.', file=sys.stderr)
        return
    spec = {'deny': list(deny) if blocking else [], 'env': list(env) if blocking else [], 'allow': list(allowed), 'log': str(log), 'python': python}
    os.environ[GUARD_ENV] = json.dumps(spec)


def _uncovered_spawns(rows: list[dict[str, Any]]) -> set[str]:
    """The ``spawned`` rows no guard covers: a non-Python command, or a Python one that cannot arm (named with why)."""
    uncovered = {str(r.get('command')) for r in rows if not r.get('python')}
    return uncovered | {f'{r.get("command")} ({r.get("cannot_arm")})' for r in rows if r.get('python') and r.get('cannot_arm')}


def _egress_summary(out: Path) -> dict[str, Any]:
    """``run_summary.egress``: the preflight's verdict plus, when the guard was armed, what it saw at run time."""
    summary = dict(_LAST_EGRESS_RECORD)
    if not summary or not (spec := os.environ.get(GUARD_ENV)):
        return summary
    try:
        rows = [json.loads(line) for line in (out / EGRESS_OBSERVED_FILENAME).read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        rows, unreadable = [], exc
    else:
        unreadable = None
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_kind.setdefault(str(row.get('kind')), []).append(row)
    if json.loads(spec).get('python') is False:
        # No guard here can observe anything; the SDK's payload-codec notes are written by the Python child of any harness.
        summary['runtime'] = {'not_covered': 'non-Python harness', **_payload_codec_notes(by_kind)}
        return summary
    if unreadable is not None:
        summary['runtime'] = {'error': f'{EGRESS_OBSERVED_FILENAME} unreadable: {type(unreadable).__name__}: {unreadable}'}
        return summary

    def _distinct(kind: str, key: str) -> list[dict[str, Any]]:
        rows = {(str(r.get(key)), str(r.get('callsite')), str(r.get('scope'))) for r in by_kind.get(kind, [])}
        return [{key: k, 'callsite': c, 'scope': sc} for k, c, sc in sorted(rows)]

    armed = [str(r.get('scope')) for r in by_kind.get('armed', [])]
    spawned = _distinct('spawned', 'command')
    # A Python subprocess is covered ONLY by its own `armed` row; one that cannot arm is named.
    python_spawns = [r for r in by_kind.get('spawned', []) if r.get('python')]
    not_covered = _uncovered_spawns(by_kind.get('spawned', []))
    if any(not r.get('cannot_arm') for r in python_spawns) and not armed.count('python-subprocess'):
        not_covered.add('python (spawned, never reported arming)')
    not_covered |= {str(r.get('why')) for r in by_kind.get('not_covered', [])}
    runtime: dict[str, Any] = {
        'guard_armed': armed.count('in-process'),
        # WHICH coverage applied: processes that REPORTED arming, by scope, and what no guard saw.
        'coverage': {
            'in-process': armed.count('in-process'),
            'python-subprocess': armed.count('python-subprocess'),
            'not_covered': sorted(not_covered),
        },
        'observed_hosts': sorted({str(r.get('host')) for r in by_kind.get('observed', [])}),
        'blocked': _distinct('blocked', 'host'),
        'spawned': spawned,
        'env_hosts': list({(r.get('env'), r.get('host')): {k: r.get(k) for k in ('env', 'host', 'blocked')} for r in by_kind.get('env_host', [])}.values()),
    }
    # Every observed host nothing named: not declared, not allow-listed (mock, backend, consent), not an
    # env-sourced endpoint. These are what the agent reached on its own — a library fetching a price
    # list included — and the preflight's source scan cannot see them. Read off the socket record, so
    # no source-scan junk filter applies: an observed host is real by construction.
    armed_spec = json.loads(spec)
    named = {*_LAST_EGRESS_DECLARED, *(str(h) for h in armed_spec.get('allow') or ()), *(str(h) for h in armed_spec.get('deny') or ())}
    named |= {str(r.get('host')) for r in by_kind.get('env_host', []) if r.get('host')}
    runtime['undeclared_observed'] = [h for h in runtime['observed_hosts'] if h.lower() not in {n.lower() for n in named}]
    if blocked_rows := by_kind.get('blocked'):
        runtime['remedy'] = blocked_rows[0].get('remedy')
    # Keyed on the guard being ARMED with something to deny (the Python guard is the enforcement layer),
    # never on the verdict: a crashed or undecided measurement leaves the guard just as armed.
    guard_blocks = summary.get('enforcement') == 'python_guard' and bool(armed_spec.get('deny') or armed_spec.get('env'))
    if guard_blocks and not runtime['guard_armed']:
        runtime['coverage_gap'] = {'gap': 'never_armed', 'commands': []}
        runtime['warning'] = 'the in-process guard was armed but no agent process recorded arming it; the declared hosts were not blocked in-process.'
    elif guard_blocks and (started := _uncovered_spawns([r for r in by_kind.get('spawned', []) if r.get('scope') == 'in-process'])):
        runtime['coverage_gap'] = {'gap': 'uncovered_spawn', 'commands': sorted(started)}
        runtime['warning'] = (
            f'the in-process guard is armed, but the agent process started {", ".join(sorted(started))}, which the guard does not cover: a declared '
            'host reached from them was neither probed nor blocked. If the agent runs there, declare `egress.agent_runtime: subprocess`.'
        )
    runtime.update(_payload_codec_notes(by_kind))
    summary['runtime'] = runtime
    return summary


#: The payload-codec notes the SDK appends to the guard's record (``plurai._mocking.codecs``), by row kind → summary key.
_PAYLOAD_CODEC_NOTES: dict[str, tuple[str, tuple[str, ...]]] = {
    'undeclared_encoded_payload': ('undeclared_encoded_payloads', ('tool', 'path', 'guess')),
    'payload_codec_failed': ('payload_codec_failures', ('tool', 'path', 'direction', 'reason')),
    'payload_codec_passthrough': ('payload_codec_passthroughs', ('tool', 'path')),
}


def _payload_codec_notes(by_kind: dict[str, list[dict[str, Any]]]) -> dict[str, list[Any]]:
    """Each payload-codec note kind the run recorded, de-duplicated across processes; no key when there were none.
    Advisory: a field the wire carries encoded that the mock read encoded (or could not decode). Never the value."""
    notes: dict[str, list[Any]] = {}
    if cut := sorted({str(r.get('suppressed')) for r in by_kind.get(SUPPRESSED_KIND, [])} & set(_PAYLOAD_CODEC_NOTES)):
        notes[SUPPRESSED_KIND] = cut  # a note kind hit its per-process cap: the lists below are a sample, not the whole
    for kind, (key, fields) in _PAYLOAD_CODEC_NOTES.items():
        distinct = {tuple(_plain(str(r.get(f))) for f in fields) for r in by_kind.get(kind, [])}
        if distinct:
            notes[key] = [dict(zip(fields, values, strict=True)) for values in sorted(distinct)]
    return notes


def _report_egress_runtime(out: Path) -> None:
    """What the in-process guard did during the run, as gate findings — once, after the run.

    Each distinct blocked host is ``egress.guard_blocked`` (SECURITY: the refusal is the isolation, so an
    overrule naming it is refused and the deny list never reads one), and a ``guarded`` verdict the run
    could not fully back is ``egress.guard_coverage_gap``. Read off the same guard log the summary folds.
    """
    runtime = _egress_summary(out).get('runtime')
    if not isinstance(runtime, dict):
        return
    gates = current_gate_authority()
    for row in runtime.get('blocked') or ():
        gates.emit(
            make_finding(
                EGRESS_GUARD_BLOCKED,
                subject=str(row.get('host')),
                evidence={'host': str(row.get('host')), 'callsite': str(row.get('callsite')), 'scope': str(row.get('scope'))},
                observed={},
                remedy=str(runtime.get('remedy') or 'Route that host through the mock; consent to reach it live is `egress.allow_hosts`.'),
                draft_ref=None,
            )
        )
    if isinstance(gap := runtime.get('coverage_gap'), dict):
        gates.emit(
            make_finding(
                EGRESS_GUARD_COVERAGE_GAP,
                subject='egress',
                evidence={'gap': str(gap.get('gap')), 'commands': list(gap.get('commands') or [])},
                observed={},
                remedy=str(runtime.get('warning')),
                draft_ref=None,
            )
        )


def _bench_repo(cfg: Any) -> Path:
    """The root above the harness — the rung :func:`plurai.host_effects.scrub_parent_record` reads as allowed."""
    return Path(os.path.normpath(Path(cfg.agent_module_path).absolute())).parent.parent


def _record_egress(out: Path, record: dict[str, Any], *, repo: Path) -> None:
    """File the egress verdict in BOTH channels: its own artifact and the module handoff that
    puts a summary of it into ``run_summary.json``.

    Two channels because they answer different questions. The file is written even when the run
    is REFUSED and no verdict exists; the summary key is what every existing consumer of a
    finished run already reads. A verdict in only the first is a verdict nothing consults.

    Both carry the record as it may leave the machine: its paths (the scan root, the config, the
    sources a host was found in) from ``~`` and never naming the operator.
    """
    record = scrub_parent_record(record, repo=repo)
    _LAST_EGRESS_RECORD.clear()
    _LAST_EGRESS_DECLARED.clear()
    _LAST_EGRESS_DECLARED.update(str(h.get('host')) for h in (record.get('declared') or ()) if isinstance(h, dict) and h.get('host'))
    kept = ('status', 'probe_mode', 'enforcement', 'detail', 'guard_scope', 'deny_hosts', 'not_determinable_cause')
    _LAST_EGRESS_RECORD.update({k: record.get(k) for k in kept if k in record})
    if record.get('discovery'):
        _LAST_EGRESS_RECORD['discovery'] = record['discovery']
    # DISTINCT hosts, not rows. A declaration is expanded to one row per probed port
    # so counting rows would report one host declared on 80 and 443 as two.
    # Only when the record actually CARRIES them. The synthetic records (an unusable declaration,
    # a crashed preflight) have no host lists at all, and writing `declared_hosts: 0` there states
    # a measurement — "we looked and found none declared" — that nothing performed.
    if 'declared' in record:
        _LAST_EGRESS_RECORD['declared_hosts'] = len({h.get('host') for h in (record['declared'] or ()) if isinstance(h, dict)})
    if 'undeclared' in record:
        _LAST_EGRESS_RECORD['undeclared_hosts'] = len(record['undeclared'] or ())
    # WHERE the "no other host" half of the verdict was measured, and WHO said so, so a reader of
    # ``run_summary.json`` (not only of ``egress_preflight.json``) can tell a root the operator
    # declared from one the marker walk found. Same `in record` guard as the two counts above — the synthetic records (an
    # unusable declaration, a crashed preflight) resolved no root at all, and writing one there
    # would state a measurement nothing performed.
    for key in ('scan_root', 'scan_root_source', 'scan_files_read'):
        if key in record:
            _LAST_EGRESS_RECORD[key] = record[key]
    if record.get('warnings'):
        _LAST_EGRESS_RECORD['warnings'] = list(record['warnings'])
    if record.get('live_by_consent'):
        _LAST_EGRESS_RECORD['live_by_consent'] = list(record['live_by_consent'])
    _write_egress_record(out, record)


def _write_egress_record(out: Path, record: dict[str, Any]) -> None:
    """Persist the preflight beside the run artifacts. Best-effort, like every other report
    writer here: a record ABOUT a run must never BE the run's failure — but a run claiming
    isolation with no evidence anywhere is exactly what this file exists to prevent, so a failed
    write reaches stderr and not only the logger."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / EGRESS_PREFLIGHT_FILENAME, record)
    except Exception as exc:  # noqa: BLE001
        logger.warning('could not write %s to %s', EGRESS_PREFLIGHT_FILENAME, out, exc_info=True)
        print(
            f'[egress] WARNING: could not write {EGRESS_PREFLIGHT_FILENAME} ({type(exc).__name__}: {exc}) — this run carries no egress evidence.',
            file=sys.stderr,
        )


#: The drift re-check's own record, beside ``egress_preflight.json`` and for the same reason:
#: a statement about the MACHINE the run happened on, written even when the run is refused.
SCAN_CONTRACT_RECHECK_FILENAME = 'scan_contract_recheck.json'

#: The last drift verdict this process computed, so ``run_summary.json`` can carry it — the same
#: module-level handoff ``_LAST_EGRESS_RECORD`` uses, for the same reason (the preflight runs
#: before the strategies and the summary is written after them, with no frame spanning both).
_LAST_SCAN_DRIFT_RECORD: dict[str, Any] = {}


def _preflight_scan_contract(cfg: _RunConfig) -> GateFinding | None:
    """Are the sources this suite was built over still the sources on this machine?

    Returns the ``run.scan_source_drift`` finding for a FULL run whose corpus MEASURABLY changed —
    reported to the run's gate collector here, and held by the one exit funnel until an
    ``--overrule`` lifts it — else ``None``. A FULL run that WAS given a baseline but whose comparison
    could not be made reports ``run.scan_drift_unchecked`` (typed ``reason_code``) instead: holding,
    because a check that was asked for and did not happen must never read as a clean one.

    It re-hashes ``--scan-corpus`` roots and compares them against the ``source_inventory``
    recorded in the ``--scan-contract`` baseline a previous ``plurai-scan-contract check`` wrote.
    Three outcomes, kept three:

    * MATCHED — the suite's premises still describe the files.
    * DRIFTED — they no longer do. The scenarios and evals were authored over documents that
      have since changed, so a final run is about to score an agent against a stale premise.
    * NOT CHECKED — no baseline was passed, or one was passed and could not be used. Never
      folded into ``matched`` and never a drift verdict. "Nobody asked" is no finding; "asked, and
      the comparison could not be made" is ``run.scan_drift_unchecked`` on a full run.

    Only a FULL run holds: qualification is the pass that finds the staleness and rebuilds what it
    touches. Every outcome is recorded either way — ``scan_contract_recheck.json`` and the
    run-summary key carry it.
    """
    if cfg.scan_contract is None:
        # Said out loud, once, exactly as `not_declared` is: a check that silently reports
        # nothing reads exactly like a check that found nothing.
        _record_scan_drift(cfg.out, {'checked': False, 'summary': 'corpus source drift NOT checked: no --scan-contract baseline was passed to this run'})
        print(
            "[scan-contract] NOT CHECKED: no --scan-contract baseline was passed, so nothing verified that this suite's "
            'corpus is still the corpus on this machine. Run `plurai-scan-contract check` and pass its scan_contract.json.',
            file=sys.stderr,
        )
        return None
    outcome_record = _scan_drift_outcome(cfg, cfg.scan_contract)
    message = str(outcome_record['summary'])
    _record_scan_drift(cfg.out, outcome_record)
    # THREE labels for three states, spelled out rather than nested: ``checked`` and ``matched``
    # are separate facts (a comparison that never happened has no verdict to report), and the
    # NOT-CHECKED rung must not be reachable by falling off the end of a MATCH/DRIFT pair.
    checked = bool(outcome_record.get('checked'))
    drift = outcome_record.get('drift') or {}
    matched = bool(drift.get('matched'))
    if matched:
        label = 'MATCH'
    elif not checked:
        label = 'NOT CHECKED'
    else:
        label = 'DRIFT'
    print(f'[scan-contract] {label}: {message}', file=sys.stderr)
    if cfg.run_phase != 'full':
        return None
    gates = current_gate_authority()
    # A baseline was passed, so both gates were evaluated: an overrule of either reads `stale`, never `not_evaluated`.
    gates.ran(RUN_SCAN_SOURCE_DRIFT)
    gates.ran(RUN_SCAN_DRIFT_UNCHECKED)
    if not checked:
        gates.emit(
            make_finding(
                RUN_SCAN_DRIFT_UNCHECKED,
                subject='corpus_sources',
                evidence={'reason_code': str(outcome_record['reason_code'])},
                observed={'baseline': str(cfg.scan_contract), 'roots': [str(root) for root in cfg.scan_corpus], 'summary': message},
                remedy=f'{message}. Pass the --scan-corpus roots the baseline was recorded over (and, for a baseline that recorded '
                'no scope, --scan-suffixes at the scope it was taken at), '
                're-record the baseline with `plurai-scan-contract check`, or rebuild the suite over a replaced corpus.',
                draft_ref=None,
            )
        )
        return None
    if matched:
        return None
    finding = make_finding(
        RUN_SCAN_SOURCE_DRIFT,
        subject='corpus_sources',
        evidence={
            'recorded_hash': str(drift.get('recorded_hash') or ''),
            'current_hash': str(drift.get('current_hash') or ''),
            'added': sorted(drift.get('added') or ()),
            'removed': sorted(drift.get('removed') or ()),
            'changed': sorted(drift.get('changed') or ()),
        },
        observed={'baseline': str(cfg.scan_contract), 'roots': [str(root) for root in cfg.scan_corpus], 'summary': message},
        remedy=f'the corpus this suite was built over is not the corpus on this machine: {message}. Rebuild what the changed sources '
        'touch and re-run `plurai-scan-contract check` to record a fresh baseline.',
        draft_ref=None,
    )
    gates.emit(finding)
    return finding


def _scan_drift_outcome(cfg: _RunConfig, baseline_path: Path) -> dict[str, Any]:
    """The drift comparison as a record, or the reason it could not be made.

    ``checked`` is the field that decides everything downstream, and the three ways a comparison
    fails to happen are kept here rather than at the call site so none of them can leak into a
    drift verdict:

    * no ``--scan-corpus`` root — the check was asked for with nothing to re-hash;
    * a root that does not exist — ``os.walk`` yields nothing for a path that does not exist
      and never raises, so a typo'd root produces an EMPTY inventory, which compares as every
      recorded file removed and reads as total drift. A manufactured verdict from a failed
      measurement is exactly what this check must never report;
    * the sides sharing no path at all (:attr:`RecheckOutcome.shares_no_path`) — different roots,
      the multi-root path prefix, or a wholesale replacement, indistinguishable from here;
    * the two walks disagreeing about their SUFFIX SCOPE. A baseline recorded with
      ``--suffixes html`` that is re-hashed at the default scope reports every one of those pages
      as ``removed`` with the files untouched — a manufactured DRIFTED verdict in the
      customer-facing Run-integrity section. ``shares_no_path`` cannot catch it: the removal is
      partial. The scope therefore travels on the inventory and the re-check walks at the scope
      the baseline RECORDED; ``--scan-suffixes`` says which scope the baseline was taken at only
      for a baseline that recorded none.
    """
    baseline = str(baseline_path)
    roots = [str(root) for root in cfg.scan_corpus]

    def not_checked(reason_code: str, reason: str) -> dict[str, Any]:
        """One NOT-CHECKED record. ``checked: False``, the typed ``reason_code`` (the closed set
        :data:`~plurai.gates.catalogue.RUN_SCAN_DRIFT_UNCHECKED` documents) and the leading phrase are fixed HERE
        so no branch below can ship a reason without them — they are what every reader, the run-summary
        handoff, the caller's own label and the ``run.scan_drift_unchecked`` finding tell a non-comparison by."""
        return {'checked': False, 'reason_code': reason_code, 'baseline': baseline, 'roots': roots, 'summary': f'corpus source drift NOT checked: {reason}'}

    if not cfg.scan_corpus:
        return not_checked('no_corpus_root', '--scan-contract was passed with no --scan-corpus root to re-hash')
    if missing := [str(root) for root in cfg.scan_corpus if not root.exists()]:
        return not_checked('corpus_root_missing', f'--scan-corpus root(s) do not exist on this machine: {", ".join(missing)}')
    try:
        # Read first so an unusable baseline gets its own reason code; the re-check reads it again and agrees.
        read_recorded_inventory(baseline_path)
    except BadUsageError as exc:
        return not_checked('baseline_unusable', str(exc))
    except Exception as exc:  # noqa: BLE001 — any re-check fault is reported as NOT CHECKED, never as drift
        return not_checked('recheck_raised', f'the baseline read raised ({type(exc).__name__}: {exc})')
    try:
        outcome = recheck_source_drift(list(cfg.scan_corpus), baseline_path, suffixes=admitted_suffixes(cfg.scan_suffixes))
    except Exception as exc:  # noqa: BLE001 — any re-check fault is reported as NOT CHECKED, never as drift
        return not_checked('recheck_raised', f'the re-check raised ({type(exc).__name__}: {exc})')
    if not outcome.checked:
        # The baseline read above, so the comparison was refused (a scope it abstains on: the recorded population is
        # in hand) or the corpus could not be re-hashed (no population on either side).
        code = 'comparison_blocked' if outcome.recorded_files else 'corpus_unhashable'
        return {**outcome.as_record(), 'reason_code': code, 'roots': roots}
    if outcome.shares_no_path:
        return {
            **not_checked(
                'shares_no_path',
                f'the recorded scan and this re-check share NO file path '
                f'({outcome.recorded_files} recorded, {outcome.current_files} found now). TWO causes produce this and they are '
                'indistinguishable from here: --scan-corpus naming different roots (or a different NUMBER of roots — the unit '
                'paths are root-name-prefixed only above one root), or the corpus having been SWAPPED WHOLESALE, which is the '
                'loudest drift there is. Check the roots first; if they are right, treat this as a full corpus replacement and '
                'rebuild the suite over it',
            ),
            'drift': outcome.drift.as_dict() if outcome.drift else None,
        }
    return {**outcome.as_record(), 'roots': roots}


def _record_scan_drift(out: Path, record: dict[str, Any]) -> None:
    """File the drift verdict in BOTH channels — its own artifact and the run-summary handoff.

    The same two-channel shape :func:`_record_egress` uses, and for the same reason: the file is
    written even when the run is refused and no verdict exists, and the summary key is what every
    existing consumer of a finished run already reads.
    """
    _LAST_SCAN_DRIFT_RECORD.clear()
    _LAST_SCAN_DRIFT_RECORD.update({k: record[k] for k in ('checked', 'summary') if k in record})
    if isinstance(record.get('drift'), dict):
        _LAST_SCAN_DRIFT_RECORD['matched'] = record['drift'].get('matched')
        _LAST_SCAN_DRIFT_RECORD['content_hash'] = record['drift'].get('current_hash')
    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / SCAN_CONTRACT_RECHECK_FILENAME, record)
    except Exception as exc:  # noqa: BLE001
        logger.warning('could not write %s to %s', SCAN_CONTRACT_RECHECK_FILENAME, out, exc_info=True)
        print(f'[scan-contract] WARNING: could not write {SCAN_CONTRACT_RECHECK_FILENAME} ({type(exc).__name__}: {exc}).', file=sys.stderr)


#: The last solvability census this process took, for ``run_summary.json`` — same module-level
#: handoff, same reason, as ``_LAST_EGRESS_RECORD``.
_LAST_SOLVABILITY_RECORD: dict[str, Any] = {}

#: The scenario SET's own shape, read off the same fetch: scenarios requested vs shipped, and the
#: steering demands it delivered short. ``{}`` means the set was not read.
_LAST_SCENARIO_SET_RECORD: dict[str, Any] = {}
#: Each scenario's ``path_probability`` as the set served it (``None`` when it carries none — a replay), off the
#: same fetch, for ``run_summary.json`` → :data:`plurai.scoring.LIKELIHOODS_KEY`. ``{}`` means the set was not read.
_LAST_SCENARIO_LIKELIHOODS: dict[str, Any] = {}


def _scenario_set_record(fetched: Any) -> dict[str, Any]:
    """``requested`` / ``shipped`` / ``undelivered_demands`` off a fetched set. Total: a malformed
    record yields ``requested: None`` or no demand rows, never a raise."""
    meta = getattr(fetched, 'dataset_metadata', None)
    meta = meta if isinstance(meta, dict) else {}
    requested = meta.get('total_scenarios')
    rows = meta.get('steering_demands')
    rows = rows if isinstance(rows, list) else []
    short = [
        {key: row.get(key) for key in ('kind', 'subject', 'slots', 'delivered')}
        for row in rows
        if isinstance(row, dict) and isinstance(row.get('slots'), int) and isinstance(row.get('delivered'), int) and row['delivered'] < row['slots']
    ]
    return {
        'requested': requested if isinstance(requested, int) else None,
        'shipped': len(getattr(fetched, 'scenarios', None) or []),
        'undelivered_demands': short,
    }


#: Every DISTINCT backend stamp this run's finalize responses carried, in first-seen order —
#: same module-level handoff, same reason, as its two neighbours (the stamps arrive per scenario
#: inside the strategies; the summary is written after them). The stamps bind a verdict to the
#: backend that produced it; ``build_id``, the one other field that travels, is a label baked
#: into the image that a source patch does not move.
_BACKEND_STAMPS: list[dict[str, Any]] = []


#: Per row the backend scored over a world it does not certify, why — same module-level handoff,
#: same reason. Its own map because ``finalize.status`` never says it: ``'failed'`` means a decisive
#: policy fired, and a reader that took it for "the world is untrustworthy" would print
#: "NEVER SCORED" over a red with landed cells.
_UNTRUSTWORTHY_ROWS: dict[str, str] = {}

#: Per row, the host-effects record its harness reported on ``agent_state``: which
#: host-writing or external tools the operator granted and which the harness denied, and how.
#: Same module-level handoff as its neighbours; the summary rolls it up under the same key.
_HOST_EFFECTS_ROWS: dict[str, dict[str, Any]] = {}

#: Per row whose tool channel went UNREAD on a bench that declares ``agent.native_tools`` (its
#: ``tool_events`` is ``None``), those declared tools (:func:`_note_tool_channel`). Same handoff.
_TOOL_CHANNEL_UNREAD_ROWS: dict[str, list[str]] = {}

#: Per row whose agent was served a stub for a tool call the mock never answered
#: (:func:`_note_mock_unanswered`): ``{calls, restart_shaped, served_schema_stub}``. Same handoff.
_MOCK_UNANSWERED_ROWS: dict[str, dict[str, int]] = {}

#: Per row whose child CLAIMED NOTHING about unanswered tool calls (``unanswered_mock_calls is None``: a
#: ``child_factory`` peer, a reaped child, no session) — ``True`` each (:func:`_note_mock_unanswered`). Not a
#: breach and never clean: a channel nobody measured (``measurement_unchecked_because: mock_unanswered``).
_MOCK_UNANSWERED_UNCLAIMED_ROWS: dict[str, bool] = {}

#: Per row whose agent replies the child TRIMMED to fit one IPC line, how many turns (:func:`_note_ipc_trimmed`).
#: A breach of the measurement — those turns' tool trail never reached the judges. Same handoff.
_IPC_TRIMMED_ROWS: dict[str, int] = {}


@dataclass(slots=True)
class _CreditWaitTally:
    """What one run's credit waits cost, for ``run_summary.json``: how many scenarios waited for room, how many of their
    starts were refused on the way, and the seconds they spent waiting (summed over the waiting scenarios)."""

    scenarios: set[str] = dataclasses_field(default_factory=set)
    refusals: int = 0
    seconds: float = 0.0

    def clear(self) -> None:
        self.scenarios.clear()
        self.refusals = 0
        self.seconds = 0.0

    def to_record(self) -> dict[str, int] | None:
        """The ``credit_waits`` block, or ``None`` when no start waited."""
        if not self.scenarios:
            return None
        return {'scenarios': len(self.scenarios), 'refusals': self.refusals, 'seconds': round(self.seconds)}


_CREDIT_WAIT_TALLY = _CreditWaitTally()


@dataclass(frozen=True, slots=True)
class _RunLedgers:
    """The per-ROW module ledgers of ONE run, as one set: the scenario-keyed facts a run's rows
    file while they run and its summary reads after (backend stamps, untrustworthy worlds, host
    effects, quarantined and harness-faulted rows).

    A ``--trials N`` invocation runs N runs CONCURRENTLY in one process, and these are keyed by
    scenario id — the same ids in every trial — so a process-wide ledger would let trial 2's row
    land in trial 1's summary. Each trial body therefore installs its OWN set in
    :data:`_TRIAL_LEDGERS` before it gathers; the tasks it spawns inherit it. A single-run
    invocation installs none and reads the module globals.
    """

    backend_stamps: list[dict[str, Any]]
    untrustworthy: dict[str, str]
    host_effects: dict[str, dict[str, Any]]
    quarantined: set[str]
    not_run: set[str]
    harness_fault: set[str]
    tool_channel_unread: dict[str, list[str]]
    mock_unanswered: dict[str, dict[str, int]]
    mock_unanswered_unclaimed: dict[str, bool]
    ipc_trimmed: dict[str, int]
    credit_waits: _CreditWaitTally

    @classmethod
    def fresh(cls) -> _RunLedgers:
        return cls(
            backend_stamps=[],
            untrustworthy={},
            host_effects={},
            quarantined=set(),
            not_run=set(),
            harness_fault=set(),
            tool_channel_unread={},
            mock_unanswered={},
            mock_unanswered_unclaimed={},
            ipc_trimmed={},
            credit_waits=_CreditWaitTally(),
        )


#: The ledger set of the TRIAL this task belongs to; ``None`` outside a trial body.
_TRIAL_LEDGERS: ContextVar[_RunLedgers | None] = ContextVar('plurai_trial_ledgers', default=None)


def _run_ledgers() -> _RunLedgers:
    """The per-row ledgers every reader and writer goes through: the current trial's, else the module's."""
    if (ledgers := _TRIAL_LEDGERS.get()) is not None:
        return ledgers
    return _RunLedgers(
        backend_stamps=_BACKEND_STAMPS,
        untrustworthy=_UNTRUSTWORTHY_ROWS,
        host_effects=_HOST_EFFECTS_ROWS,
        quarantined=_QUARANTINED_ROWS,
        not_run=_NOT_RUN_ROWS,
        harness_fault=_HARNESS_FAULT_ROWS,
        tool_channel_unread=_TOOL_CHANNEL_UNREAD_ROWS,
        mock_unanswered=_MOCK_UNANSWERED_ROWS,
        mock_unanswered_unclaimed=_MOCK_UNANSWERED_UNCLAIMED_ROWS,
        ipc_trimmed=_IPC_TRIMMED_ROWS,
        credit_waits=_CREDIT_WAIT_TALLY,
    )


@dataclass(frozen=True, slots=True)
class _MockAbort:
    """Why a run stopped: a hosted mock could not answer, and every try and every scenario would get the same answer.

    Raised by the first scenario to meet a cause in :data:`ABORTING_ROUTE_CAUSES` (``auth``, ``not_deployed``,
    ``not_a_mock``). The agent would otherwise be scored on tools that never worked, and the score read as a real one.
    ``tool`` and ``status_code`` are ``None`` when the cause arrived as a typed turn fault, which names neither.
    """

    cause: MockRouteCause
    scenario_id: str
    remedy: str
    tool: str | None = None
    status_code: int | None = None

    def to_record(self) -> dict[str, Any]:
        """The ``aborted`` block of ``run_summary.json``."""
        return {
            'reason': 'mock_unanswered',
            'cause': self.cause,
            'scenario': self.scenario_id,
            'tool': self.tool,
            'status_code': self.status_code,
            'remedy': self.remedy,
        }

    @property
    def cause_phrase(self) -> str:
        """The cause in the words of the ``[abort]`` line."""
        return f'the mock cannot answer ({self.cause})'

    @property
    def fail_record(self) -> tuple[str, str]:
        """The ``(exception, error)`` a cancelled scenario's thread is failed with."""
        return 'MockUnanswered', 'the run was stopped because a mock could not answer'


@dataclass(frozen=True, slots=True)
class _CreditAbort:
    """Why a run stopped: the backend refused work for credit, and every further call would be refused the same way.

    Raised by the first scenario to meet a credit refusal. Unlike :class:`_MockAbort` it keeps what the run measured: the
    scenarios that finished keep their artifacts and verdicts, the rest are recorded as not run. ``remedy`` is the kind's
    sentence (:func:`plurai.credit.credit_sentence`), printed once when the run ends and written into the summary.
    """

    code: CreditKind
    scenario_id: str | None
    remedy: str
    resets_at: str | None = None
    route_cause: ClassVar[MockRouteCause] = 'credit'

    def to_record(self) -> dict[str, Any]:
        """The ``aborted`` block of ``run_summary.json``; ``resets_at`` is absent, never ``null``, when unknown."""
        record: dict[str, Any] = {'reason': 'credit', 'code': self.code}
        if self.resets_at is not None:
            record['resets_at'] = self.resets_at
        if self.scenario_id is not None:
            record['scenario'] = self.scenario_id
        record['remedy'] = self.remedy
        return record

    @property
    def cause_phrase(self) -> str:
        """The cause in the words of the ``[abort]`` line."""
        return _stop_cause(self.code, with_code=True)

    @property
    def fail_record(self) -> tuple[str, str]:
        """The ``(exception, error)`` a cancelled scenario's thread is failed with."""
        return 'CreditRefused', 'the run was stopped because credit was refused'

    @property
    def stopped_by(self) -> str:
        """What stopped the run, in the words of its run-end line."""
        return _stop_cause(self.code)


#: What a cancelled scenario's thread is failed with when the backend refused the run (capacity or unavailable): the same
#: words as the :class:`BackendUnavailable` record in ``error.json``, which names no status, URL or id.
_BACKEND_REFUSED_ERROR = f'the run was stopped because {BACKEND_REFUSED_PHRASE}'


@dataclass(frozen=True, slots=True)
class _BackendRefusedAbort:
    """Why a run stopped: the backend refused its work (capacity) or could not take it (unavailable), and every further call
    would fail the same way for now.

    The shape of :class:`_CreditAbort`: raised by the first scenario to meet the refusal, it keeps what the run measured (the
    scenarios that finished keep their artifacts and verdicts, the rest are recorded as not run), and every conversation it
    cancels is reported to ``/fail``. A scored run is never played on into refusals. ``remedy`` is the stop's one text
    (:data:`plurai.capacity.CAPACITY_TEXT` or :data:`plurai.capacity.UNAVAILABLE_TEXT`), printed once when the run ends and
    written into the summary. No customer line names more than :data:`~plurai.capacity.BACKEND_REFUSED_PHRASE`; a subclass
    adds the machine fields its record carries.
    """

    reason: ClassVar[str]
    route_cause: ClassVar[MockRouteCause]
    #: The exception name a cancelled scenario's ``/fail`` carries.
    fail_exception: ClassVar[str] = 'BackendRefused'

    scenario_id: str | None
    remedy: str

    def _record_fields(self) -> dict[str, Any]:
        """The fields of the ``aborted`` block that follow ``reason``; a subclass adds its own."""
        return {}

    def to_record(self) -> dict[str, Any]:
        """The ``aborted`` block of ``run_summary.json``."""
        record: dict[str, Any] = {'reason': self.reason, **self._record_fields()}
        if self.scenario_id is not None:
            record['scenario'] = self.scenario_id
        record['remedy'] = self.remedy
        return record

    @property
    def cause_phrase(self) -> str:
        """The cause in the words of the ``[abort]`` line."""
        return BACKEND_REFUSED_PHRASE

    @property
    def fail_record(self) -> tuple[str, str]:
        """The ``(exception, error)`` a cancelled scenario's thread is failed with."""
        return self.fail_exception, _BACKEND_REFUSED_ERROR

    @property
    def stopped_by(self) -> str:
        """What stopped the run, in the words of its run-end line."""
        return BACKEND_REFUSED_PHRASE


@dataclass(frozen=True, slots=True)
class _CapacityAbort(_BackendRefusedAbort):
    """The backend refused work because the provider refused its account.

    ``resets_at`` / ``reset_source`` are the backend's block, recorded (absent, never ``null``, when unknown) so a reader renders
    the stop from the record alone; the capacity text never reads a reset time.
    """

    reason: ClassVar[str] = 'capacity'
    route_cause: ClassVar[MockRouteCause] = 'capacity'

    resets_at: str | None = None
    reset_source: str | None = None

    def _record_fields(self) -> dict[str, Any]:
        return CapacityBlock(self.resets_at, self.reset_source).to_record()


@dataclass(frozen=True, slots=True)
class _UnavailableAbort(_BackendRefusedAbort):
    """The backend could not be reached (a transient failure outlasted the retry budget, and the re-drive's readiness gate
    or the re-drive itself met it again) or said it is unavailable (its typed ``unavailable`` block on its own answer).

    ``cause`` is where the stop came from (``transport``, ``status`` or ``backend``). ``code`` is the backend's own, recorded
    when the stop came from a typed block that named one and absent for a transport cause; no status or id rides with it.
    """

    reason: ClassVar[str] = 'unavailable'
    route_cause: ClassVar[MockRouteCause] = 'unavailable'
    fail_exception: ClassVar[str] = 'BackendUnavailable'

    cause: UnavailableCause
    code: str | None = None

    def _record_fields(self) -> dict[str, Any]:
        return UnavailableBlock(self.cause, self.code).to_record()


class BackendUnavailable(Exception):  # noqa: N818  # named for the stop it records, not as an error of ours
    """The failure record of a scenario the unavailable stop ended: what ``error.json`` and ``/fail`` carry in place of the
    transport's own text, which names a status, a URL or an id. It has the name the cancelled scenarios' ``/fail`` carries
    (:attr:`_UnavailableAbort.fail_record`); ``unavailable`` is the block the record reads."""

    def __init__(self, block: UnavailableBlock) -> None:
        super().__init__(_BACKEND_REFUSED_ERROR)
        self.unavailable = block


#: The stops that keep what the run measured: a refusal of the organization's credit, of the backend's capacity, or the
#: backend being unavailable.
_MeasuredStop = _CreditAbort | _CapacityAbort | _UnavailableAbort

#: Every way a run can be stopped as a whole.
_RunAbort = _MockAbort | _CreditAbort | _CapacityAbort | _UnavailableAbort

#: The one abort of the running invocation, shared by every trial and both strategies: the mock and the organization's
#: credit are the run's, not a trial's, so a ledger per trial would let trial 2 keep going after trial 1 found them dead.
#: First writer wins.
_RUN_ABORT: _RunAbort | None = None
_RUN_ABORT_LOCK = threading.Lock()

#: How often a running scenario checks whether the run was aborted. The vendor executor of the experiment strategy
#: owns its own task loop, so the check polls a flag instead of waiting on an event bound to one loop.
_ABORT_POLL_SECONDS = 0.2


def _run_abort() -> _RunAbort | None:
    """The run's abort, or ``None`` while the run may go on."""
    return _RUN_ABORT


def _credit_abort() -> _CreditAbort | None:
    """The run's abort when it is a credit refusal, else ``None``."""
    abort = _RUN_ABORT
    return abort if isinstance(abort, _CreditAbort) else None


def _measured_stop() -> _MeasuredStop | None:
    """The run's abort when it keeps what the run measured (a credit or a capacity refusal, or an unavailable backend), else ``None``."""
    abort = _RUN_ABORT
    return abort if isinstance(abort, _MeasuredStop) else None


def _request_run_abort(abort: _RunAbort) -> bool:
    """Stop the run for ``abort``; ``True`` when this call is the one that stopped it.

    The announcement names the cause and never the credit sentence: that is printed once, when the run ends.
    """
    global _RUN_ABORT  # noqa: PLW0603 — the run-wide handoff, like the ledgers above
    with _RUN_ABORT_LOCK:
        if _RUN_ABORT is not None:
            return False
        _RUN_ABORT = abort
    print(f'[{_ts()}] [abort] {abort.scenario_id}: {abort.cause_phrase}; no further scenarios start and the running ones are cancelled.')
    return True


#: How long the waiting scenario at the head of the queue holds back after an ask the gateway refused, while the run may
#: still hold room. An admitted ask holds nothing back. It also lets a run follow a balance that grows for another reason
#: (credit added, another member's run ending).
_CREDIT_PROBE_SECONDS = 60.0

#: How long a conversation that finalized still counts as holding room after the runner saw it end. The gateway keeps
#: an ended run's credit held for a short time after the run reports its end, so that its last spend reaches the credit
#: snapshot first; a start made at the instant of an end is therefore refused by design. A conversation that failed or
#: was cancelled reports no end and the gateway forgets it only after its idle timeout, which is too long to wait on:
#: it counts as gone at once.
_CREDIT_HOLD_AFTER_FINALIZE_SECONDS = 240.0


class _CreditWaits:
    """Which of the run's own conversations may still hold gateway room, and which refused scenario asks next.

    The gateway admits a conversation only while the credit left covers a margin for every run in progress, and each
    conversation is a run of its own, so a run's own first wave can fill the room for its next start. A conversation
    holds room once the gateway admitted it (its first turn landed); one that is only trying to start holds none. A
    start refused ``new-run-refused`` while the run holds room (or just released it, see
    :data:`_CREDIT_HOLD_AFTER_FINALIZE_SECONDS`) waits; with none, the refusal is about someone else's work and the run
    stops for credit.

    The refusal of a start usually arrives before the admission of a sibling that started at the same moment, so a
    refused start whose siblings are still trying decides silently: it waits for the first of their outcomes, an
    admission (it becomes a wait) or the last of them ending refused or failed (the run stops). Starts that stop the
    run decide in the order they were refused.

    One state per process: the room is the organization's, and every trial shares it. Keys are attempt tokens, not
    scenario ids, because ``--trials`` repeats the ids. The runner polls this object from its own loop and, in the
    experiment strategy, from the vendor executor's loop, so a lock guards it instead of an event bound to one loop.

    Only the waiter at the head of the queue asks, and only while no earlier ask is unresolved, so one process has at
    most one start in flight for room. An ask that is admitted lets the next waiter ask at once; one that is refused
    backs the next off by :data:`_CREDIT_PROBE_SECONDS`.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._trying: set[str] = set()
        self._open: set[str] = set()
        self._held_until: dict[str, float] = {}
        self._undecided: list[str] = []
        self._waiters: list[str] = []
        self._probing: str | None = None
        self._probe_at: float | None = None

    def reset(self) -> None:
        with self._lock:
            self._trying.clear()
            self._open.clear()
            self._held_until.clear()
            self._undecided.clear()
            self._waiters.clear()
            self._probing = None
            self._probe_at = None

    def start(self, token: str) -> None:
        """A conversation of the run begins to start; it holds no room until :meth:`admitted`."""
        with self._lock:
            self._trying.add(token)

    def admitted(self, token: str) -> None:
        """The gateway admitted the conversation of *token*: it holds room, and a waiter's ask that this resolves frees the next to ask."""
        with self._lock:
            if token not in self._trying:
                return
            self._trying.discard(token)
            self._open.add(token)
            if self._probing == token:
                self._probing = None
                if self._waiters:
                    self._probe_at = self._clock()

    def end(self, token: str, *, finalized: bool) -> None:
        """The attempt of *token* is over, however it ended; wake the head waiter when it held room.

        A conversation that finalized was admitted and keeps its room for the grace, so the head asks again once that has
        passed; one that failed is gone, so the head asks at once. An attempt that was refused, or never admitted, held no
        room: it only stops being a sibling that is trying.
        """
        with self._lock:
            if token in self._waiters:
                self._waiters.remove(token)
            if token in self._undecided:
                self._undecided.remove(token)
            held = token in self._open or finalized
            if token not in self._trying and token not in self._open:
                # The refused start of a scenario that now asks again: its ask is still in flight and ends by being
                # admitted (:meth:`admitted`) or refused (:meth:`refused`), not here.
                return
            self._trying.discard(token)
            self._open.discard(token)
            now = self._clock()
            if self._probing == token:
                # The ask ends without a refusal. A conversation that finalized was admitted, so the next waiter asks at
                # once; one that failed first says nothing about the room, so the next waiter asks on the cadence.
                self._probing = None
                if self._waiters:
                    self._reschedule(now if finalized else now + _CREDIT_PROBE_SECONDS)
            if not held:
                return
            wake = now
            if finalized:
                wake = now + _CREDIT_HOLD_AFTER_FINALIZE_SECONDS
                self._held_until[token] = wake
            if self._waiters:
                self._reschedule(wake)

    def _backed_off_until(self, now: float) -> float:
        """When the head waiter asks after a refused ask: on the cadence, or sooner when a grace ends first."""
        return min([now + _CREDIT_PROBE_SECONDS, *self._held_until.values()])

    def _reschedule(self, at: float) -> None:
        """Let the head waiter ask at *at*, unless it is already due earlier."""
        self._probe_at = at if self._probe_at is None else min(self._probe_at, at)

    def refused(self, token: str) -> Literal['wait', 'abort'] | None:
        """The start of *token* was refused for room; see :meth:`decide` for the answer."""
        with self._lock:
            self._trying.discard(token)
            self._open.discard(token)
            if self._probing == token:
                self._probing = None
                if self._waiters:
                    self._reschedule(self._backed_off_until(self._clock()))
            if token not in self._undecided:
                self._undecided.append(token)
            return self._decide(token, run_stopped=False)

    def decide(self, token: str, *, run_stopped: bool = False) -> Literal['wait', 'abort'] | None:
        """What the refused start of *token* does: ``'wait'`` while the run holds room, ``'abort'`` when it holds none and no
        sibling is still trying to start, else ``None`` (ask again later).

        *run_stopped* says the run is stopping for another reason, so no admission is worth waiting for.
        """
        with self._lock:
            return self._decide(token, run_stopped=run_stopped)

    def _decide(self, token: str, *, run_stopped: bool) -> Literal['wait', 'abort'] | None:
        if token not in self._undecided:
            return 'abort'  # the state was reset under it: the run it belonged to is over
        now = self._clock()
        self._held_until = {key: until for key, until in self._held_until.items() if until > now}
        if self._open or self._held_until:
            self._undecided.remove(token)
            if token not in self._waiters:
                self._waiters.append(token)
            if self._waiters[0] == token:
                self._probe_at = self._backed_off_until(now)
            return 'wait'
        if (self._trying and not run_stopped) or self._undecided[0] != token:
            return None
        self._undecided.pop(0)
        if token in self._waiters:
            self._waiters.remove(token)
        return 'abort'

    def try_begin_probe(self, token: str) -> bool:
        """Whether *token* may start again now; ``True`` makes it the one start in flight, until it is admitted or refused."""
        with self._lock:
            if not self._waiters or self._waiters[0] != token or self._probing is not None or self._probe_at is None:
                return False
            if self._clock() < self._probe_at:
                return False
            self._waiters.pop(0)
            self._probing = token
            self._probe_at = None
            return True


_CREDIT_WAITS = _CreditWaits()


def _refused_new_start(exc: BaseException, orch: object) -> bool:
    """Whether *exc* is the gateway refusing a conversation that has not started: ``new-run-refused`` on the create or on
    the thread's first turn (no turn landed yet).

    Every other kind, a refusal on a later turn, on a finalize or rescore job, and any credit evidence that is not an HTTP
    refusal (a hosted mock's fault) is about work already admitted or about the organization's balance, and stops the run.
    """
    if not isinstance(exc, CreditRefused) or exc.credit.code != 'new-run-refused':
        return False
    path = exc.request_path
    if path is None:
        return False
    return path.endswith('/threads') or (path.endswith('/user-turn') and not getattr(orch, 'turns_reached', []))


def _failure_text(exc: BaseException) -> str:
    """*exc* for an advisory line: a credit refusal names its kind (no status, no backend text), anything else its type and message."""
    if (block := credit_block_of(exc)) is not None:
        return f'refused for credit ({block.code})'
    return f'{type(exc).__name__}: {exc}'


def _stop_cause(code: str, *, with_code: bool = False) -> str:
    """What stopped the run, in the words the terminal uses. ``credit-unavailable`` is a fault on Plurai's side, so a user
    reads no "credit" for it; the kind stays in ``aborted.code`` and ``error.json``."""
    if code == 'credit-unavailable':
        return "a fault on Plurai's side"
    return f'credit was refused ({code})' if with_code else 'credit was refused'


def _stop_label(code: str) -> str:
    """The bracketed label of a refused scenario's failure line; the same Plurai-side split as :func:`_stop_cause`."""
    return 'STOPPED' if code == 'credit-unavailable' else 'CREDIT'


def _note_not_run(scenario_id: str) -> None:
    """File ``scenario_id`` as stopped by the abort before it could finish: an ``incomplete`` row, never a failed one."""
    _run_ledgers().not_run.add(scenario_id)


def _stop_run_if_credit_refused(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]], *, billing_url: str) -> _CreditAbort | None:
    """Stop the run when ``evidence`` is a credit refusal, and return what it shows; ``None`` for any other error.

    ``evidence`` is the error a scenario raised (an HTTP refusal, a finalize job that ended on credit, or a hosted mock's
    fault) or the scenario's ``unanswered_mock_calls``, whose rows carry a direct 402 from the mock path. The run keeps
    the first abort it was given; the one returned here is this scenario's own evidence. Whether the scenario that met it
    still counts as run is the caller's to say (:func:`_note_not_run`): a judge replay is refused after its scenario was
    graded.
    """
    found_abort = _credit_abort_from(scenario_id, evidence, billing_url=billing_url)
    if found_abort is not None:
        _request_run_abort(found_abort)
    return found_abort


def _credit_abort_from(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]], *, billing_url: str) -> _CreditAbort | None:
    """The credit abort ``evidence`` shows (see :func:`_stop_run_if_credit_refused`) without stopping anything; ``None`` for any other error."""
    if isinstance(evidence, BaseException):
        block = credit_block_of(evidence)
    else:
        block = next((found for call in evidence if isinstance(call, Mapping) and (found := credit_block_of_call(call)) is not None), None)
    if block is None:
        return None
    return _CreditAbort(
        code=block.code,
        scenario_id=scenario_id,
        remedy=credit_sentence(block.code, resets_at=block.resets_at, billing_url=billing_url),
        resets_at=block.resets_at,
    )


def _capacity_abort_from(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]]) -> _CapacityAbort | None:
    """The capacity abort ``evidence`` shows, read like :func:`_credit_abort_from`: the backend's typed capacity block on a
    hosted mock's fault, or on an ``unanswered_mock_calls`` row; ``None`` for any other error. Never read from prose."""
    if isinstance(evidence, BaseException):
        block = capacity_block_of(evidence)
    else:
        block = next((found for call in evidence if isinstance(call, Mapping) and (found := capacity_block_of_call(call)) is not None), None)
    if block is None:
        return None
    return _CapacityAbort(scenario_id=scenario_id, remedy=CAPACITY_TEXT, resets_at=block.resets_at, reset_source=block.reset_source)


def _unavailable_abort_from(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]]) -> _UnavailableAbort | None:
    """The unavailable abort ``evidence`` shows, read like :func:`_capacity_abort_from`: the block a spent transient retry budget
    left on the error, or the backend's typed ``unavailable`` block on a hosted mock's fault or an ``unanswered_mock_calls``
    row; ``None`` for any other error. Never read from prose."""
    if isinstance(evidence, BaseException):
        block = unavailable_block_of(evidence)
    else:
        block = next((found for call in evidence if isinstance(call, Mapping) and (found := unavailable_block_of_call(call)) is not None), None)
    if block is None:
        return None
    return _UnavailableAbort(scenario_id=scenario_id, remedy=UNAVAILABLE_TEXT, cause=block.cause, code=block.code)


def _stop_run_if_refused(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]], *, billing_url: str) -> _MeasuredStop | None:
    """Stop the run when ``evidence`` is a credit refusal (:func:`_stop_run_if_credit_refused`), a capacity refusal, or the
    backend's own typed ``unavailable`` answer, and return what it shows; ``None`` for any other error. Each is the same answer
    on a re-drive and for every other scenario, so the run stops on the first one and keeps what it measured; credit is read
    first.

    An outage is recovered before it stops anything, so it is not read here: a request that waited out its transient budget
    (:func:`~plurai.orchestrator.redrive.backend_outage`) and a hosted mock's unavailable answer go to the re-drive, and the
    first stops the run only once recovery failed (:func:`_stop_run_for_outage`)."""
    if (credit := _stop_run_if_credit_refused(scenario_id, evidence, billing_url=billing_url)) is not None:
        return credit
    found: _CapacityAbort | _UnavailableAbort | None = _capacity_abort_from(scenario_id, evidence)
    if found is None and isinstance(evidence, BaseException) and not _recovered_first(evidence):
        found = _unavailable_abort_from(scenario_id, evidence)
    if found is not None:
        _request_run_abort(found)
    return found


def _recovered_first(exc: BaseException) -> bool:
    """Whether *exc* is an unavailable failure the run recovers before it stops: a spent transient budget, which the re-drive's
    readiness gate waits out, or a hosted mock's unavailable answer, which fails one scenario and is re-driven like a transient
    mock fault."""
    return backend_outage(exc) or (isinstance(exc, MockRuntimeFaultRaised) and unavailable_block_of(exc) is not None)


def _stop_run_for_outage(scenario_id: str, exc: BaseException) -> _UnavailableAbort | None:
    """Stop the run as the unavailable stop for a backend outage that recovery did not cure: the backend stayed down past the
    re-drive's readiness gate, the re-drive met the outage again, or the scenario's re-drive was already spent. ``None`` when
    *exc* is no such outage (:func:`~plurai.orchestrator.redrive.backend_outage`)."""
    if not backend_outage(exc):
        return None
    found = _unavailable_abort_from(scenario_id, exc)
    if found is not None:
        _request_run_abort(found)
    return found


def _billing_url(settings: Settings) -> str:
    """The billing page of the backend ``settings`` talks to; empty when that environment has none."""
    endpoint = settings.resolved_endpoint
    return billing_url_for(endpoint.url) if endpoint is not None else ''


def _stop_run_if_mock_cannot_answer(scenario_id: str, evidence: BaseException | Sequence[Mapping[str, Any]], *, mock_url: str | None) -> _MockAbort | None:
    """Stop the run when ``evidence`` shows a mock that cannot answer, and return what it shows; ``None`` otherwise.

    ``evidence`` is the turn fault a scenario raised, or the scenario's ``unanswered_mock_calls``. A dropped transport
    and a held env are not a verdict on the mock: the first is retried, the second waits for a human. The run keeps the
    first abort it was given; the one returned here is this scenario's own evidence.
    """
    found: _MockAbort | None = None
    if isinstance(evidence, BaseException):
        cause = evidence.route_cause if isinstance(evidence, MockRuntimeFaultRaised) else None
        if cause in ABORTING_ROUTE_CAUSES:
            found = _MockAbort(cause=cause, scenario_id=scenario_id, remedy=abort_remedy(cause, mock_url))
    else:
        for call in evidence:
            if isinstance(call, Mapping) and (cause := aborting_route_cause(call)) is not None:
                tool, status = call.get('tool'), call.get('status_code')
                found = _MockAbort(
                    cause=cause,
                    scenario_id=scenario_id,
                    remedy=abort_remedy(cause, mock_url),
                    tool=tool if isinstance(tool, str) else None,
                    status_code=status if isinstance(status, int) else None,
                )
                break
    if found is not None:
        _request_run_abort(found)
    return found


class _Stopped:
    """The answer of :func:`_until_run_abort` for work the abort cancelled; ``None`` stays the answer of work that failed on its own."""


_STOPPED: Final = _Stopped()


#: The scenario tasks that a credit, capacity or unavailable stop lets finish instead of cancelling. One is polling its
#: finalize job: the failure is not its own, its judges already ran, and cancelling it would delete a verdict the
#: organization has already paid for. The poll keeps its own wallclock budget and resumes by job id through an outage, so an
#: unavailable stop spares it too. The other is deciding whether its refused start waits (:func:`_decide_refused_start`): it
#: sees the same stop as a refusal that arrives after it.
_SPARED_BY_MEASURED_STOP: set[asyncio.Task[Any]] = set()


def _mark_finalizing() -> None:
    """Record that the calling scenario task has started its finalize (the orchestrator's ``on_finalize_start``).

    A task that reaches its finalize after the run was stopped is cancelled here, before it submits one: the hold
    keeps a finalize that is already running, it does not begin another.
    """
    if _measured_stop() is not None:
        raise asyncio.CancelledError
    if (task := asyncio.current_task()) is not None:
        _SPARED_BY_MEASURED_STOP.add(task)


def _release_finalize_hold() -> None:
    """End the calling scenario task's finalize hold: the attempt it covered is over."""
    if (task := asyncio.current_task()) is not None:
        _SPARED_BY_MEASURED_STOP.discard(task)


def _hold_through_bookkeeping() -> None:
    """Keep the calling scenario task from being cancelled by the measured stop it just set, so it closes its thread and
    writes its record; the hold ends with the task, which returns right after."""
    if (task := asyncio.current_task()) is not None:
        _SPARED_BY_MEASURED_STOP.add(task)


def _releases_finalize_hold[**P, R](drive: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    """*drive* as an attempt that ends the calling task's finalize hold however it ends."""

    @functools.wraps(drive)
    async def attempt(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await drive(*args, **kwargs)
        finally:
            _release_finalize_hold()

    return attempt


async def _until_run_abort[T](work: Awaitable[T]) -> T | _Stopped:
    """``await work``, cancelled and answered with :data:`_STOPPED` the moment the run is aborted.

    Work that is polling its finalize job when the run is stopped for credit, capacity or an unavailable backend is left to
    finish (:data:`_SPARED_BY_MEASURED_STOP`), and is cancelled only once it leaves that phase with the stop still standing.
    """
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=_ABORT_POLL_SECONDS)
            if done:
                if task.cancelled() and _run_abort() is not None:
                    return _STOPPED  # the work cancelled itself because the run was already stopped
                return task.result()
            # The hold is for a measured stop: a mock abort means the scenario would be graded against stubs, so nothing
            # is worth keeping.
            if _run_abort() is not None and not (task in _SPARED_BY_MEASURED_STOP and _measured_stop() is not None):
                task.cancel()
                await asyncio.wait({task})
                if not task.cancelled():
                    task.exception()  # collected: a cleanup failure must not surface as an unretrieved exception
                return _STOPPED
    finally:
        _SPARED_BY_MEASURED_STOP.discard(task)
        if not task.done():
            task.cancel()


async def _decide_refused_start(token: str) -> Literal['wait', 'abort']:
    """Whether the start of *token*, refused for room, waits for the run's own conversations or stops the run.

    While a sibling is still trying to start the answer is not known, and this waits for it without a word. The task is
    left to finish when a credit stop arrives meanwhile (:data:`_SPARED_BY_MEASURED_STOP`): it then takes the same stop as a refusal
    that arrived after it.
    """
    task = asyncio.current_task()
    if (decision := _CREDIT_WAITS.refused(token)) is not None:
        return decision
    if task is not None:
        _SPARED_BY_MEASURED_STOP.add(task)
    try:
        while (decision := _CREDIT_WAITS.decide(token, run_stopped=_run_abort() is not None)) is None:  # noqa: ASYNC110 - a lock-guarded state the runner polls, never a loop-bound event
            await asyncio.sleep(_ABORT_POLL_SECONDS)
        return decision
    finally:
        if task is not None:
            _SPARED_BY_MEASURED_STOP.discard(task)


async def _wait_for_credit_room(orch: Any, token: str, scenario_id: str, thread_id: object) -> float | None:
    """Give back what the refused start held and wait until this scenario may start again.

    Returns the seconds waited, or ``None`` when the run was stopped meanwhile. The scenario keeps its concurrency slot:
    handing it on would start another scenario that is refused the same way, until every scenario of the run waits. The
    world lease and the setup slot are already released, and the child is closed with the orchestrator's context.
    """
    if isinstance(thread_id, str) and thread_id:
        try:
            await orch.client.fail(thread_id, exception='CreditRefused', error='the thread was refused for room and is replaced by a fresh one')
        except Exception:  # noqa: BLE001 — /fail is best-effort server bookkeeping
            logger.debug('/fail bookkeeping failed for thread %s', thread_id, exc_info=True)
    await orch._cancel_inflight_finalize()
    tally = _run_ledgers().credit_waits
    tally.refusals += 1
    if scenario_id not in tally.scenarios:
        tally.scenarios.add(scenario_id)
        print(f"[{_ts()}] [wait]  {scenario_id}: no room to start another conversation; waiting for one of this run's conversations to end")
    began = time.monotonic()
    try:
        while _run_abort() is None:
            if _CREDIT_WAITS.try_begin_probe(token):
                return time.monotonic() - began
            await asyncio.sleep(_ABORT_POLL_SECONDS)
        return None
    finally:
        tally.seconds += time.monotonic() - began


async def _watch_admission(orch: Any, token: str, scenario_id: str, waited: float | None, *, earlier_attempt: object) -> None:
    """Tell the credit state when the gateway admitted this attempt's conversation: its first turn landed.

    A scenario that waited for room says it has begun at that moment, since the start was admitted and not just tried.
    *earlier_attempt* is the turns list the orchestrator held when the attempt was created: the orchestrator starts every
    run with a fresh list, and until then the list is the last attempt's. It is taken before the task is scheduled, because
    the run replaces the list as soon as the task lets it start.
    """
    while not (turns := getattr(orch, 'turns_reached', None)) or turns is earlier_attempt:  # noqa: ASYNC110 - a list the orchestrator fills; the runner polls flags, never loop-bound events
        await asyncio.sleep(_ABORT_POLL_SECONDS)
    _CREDIT_WAITS.admitted(token)
    if waited is None:
        return
    minutes = f'{waited / 60:.0f}m' if waited >= 60 else f'{waited:.0f}s'
    try:
        print(f'[{_ts()}] [start] {scenario_id} (after waiting {minutes} for room)')
    except Exception:  # noqa: BLE001 — the announcement is not the run
        logger.debug('could not announce the start of %s', scenario_id, exc_info=True)


#: How long one bookkeeping request (``/fail``, a finalize cancel) may take once the run stopped because the backend is not
#: there: one short try, never a retry budget, so a backend that does not answer holds the stop's text back by seconds only.
_BEST_EFFORT_SECONDS = 5.0


async def _bookkeep(request: Callable[[], Awaitable[object]], *, what: str, thread_id: str | None) -> None:
    """Send one best-effort bookkeeping *request*; a failure is logged and dropped.

    While the run's stop is the unavailable one the request gets :data:`_BEST_EFFORT_SECONDS`, whatever its retry policy would
    wait. It is sent anyway: ``/fail`` is what ends the thread's run in the gateway's credit room, and a thread left open holds
    its place until the gateway's idle timeout and refuses the organization's next run.
    """
    try:
        if isinstance(_run_abort(), _UnavailableAbort):
            await asyncio.wait_for(request(), _BEST_EFFORT_SECONDS)
        else:
            await request()
    except Exception:  # noqa: BLE001 — bookkeeping is best-effort; the scenario's own outcome is already decided
        logger.debug('%s bookkeeping failed for thread %s', what, thread_id, exc_info=True)


async def _close_thread(orch: Orchestrator, thread_id: str | None, *, exception: str, error: str) -> None:
    """One best-effort ``/fail`` (:func:`_bookkeep`) of *thread_id*, then the cancel of any finalize job the attempt
    submitted; the ``/fail`` only when the attempt holds a thread."""
    if thread_id:
        await _bookkeep(lambda: orch.client.fail(thread_id, exception=exception, error=error), what='/fail', thread_id=thread_id)
    await _bookkeep(orch._cancel_inflight_finalize, what='finalize cancel', thread_id=thread_id)


async def _close_attempt_cancelled_by_abort(orch: Orchestrator, abort: _RunAbort) -> None:
    """Best-effort backend bookkeeping for a scenario the abort cancelled: close its thread once and stop any judge it
    submitted (:func:`_close_thread`)."""
    exception, error = abort.fail_record
    await _close_thread(orch, orch.thread_id, exception=exception, error=error)


def _write_mock_abort_summary(cfg: _RunConfig, abort: _MockAbort, *, run_name: str | None) -> _RunVerdict:
    """``run_summary.json`` for an aborted run: the reason and no verdicts.

    No row is passed, so no policy verdict, score or pass rate is published for a run whose mock answered nothing a score
    could rest on. The scenario artifacts already on disk stay where they are.
    """
    return _write_run_summary(cfg.out, exit_code=exit_codes.MOCK_UNANSWERED, scenarios={}, graded=set(), cfg=cfg, run_name=run_name, aborted=abort.to_record())


def _publish_mock_abort(cfg: _RunConfig, abort: _MockAbort, *, run_name: str | None) -> int:
    """The terminal side of an abort: the remedy on stderr, the summary on disk, :data:`exit_codes.MOCK_UNANSWERED` back."""
    where = f'scenario {abort.scenario_id}' + (f', tool {abort.tool}' if abort.tool else '')
    print(
        f'ERROR: the run was stopped because a mocked tool call could not be answered ({abort.cause}); no scenario was scored.\n'
        f'  first seen in {where}\n'
        f'  {abort.remedy}\n'
        f'  Exit {exit_codes.MOCK_UNANSWERED}: fix the cause above and run again.',
        file=sys.stderr,
    )
    _write_mock_abort_summary(cfg, abort, run_name=run_name)
    return exit_codes.MOCK_UNANSWERED


def _stamp_credit_abort(cfg: _RunConfig, abort: _MeasuredStop, *, run_name: str | None) -> None:
    """Put the credit (or capacity) stop into ``$OUT``'s ``run_summary.json`` of a ``--trials N`` run, so a reader finds it in one place.

    ``$OUT`` holds trial 1's own summary; a trial that finished before the refusal wrote it without an abort. The stop
    and its exit code are set on that file; a missing one is written as an aborted run with no rows.
    """
    path = cfg.out / RUN_SUMMARY_FILENAME
    try:
        summary = _read_json_object(path)
        if summary is None:
            _write_run_summary(cfg.out, exit_code=exit_codes.NOT_MEASURED, scenarios={}, graded=set(), cfg=cfg, run_name=run_name, aborted=abort.to_record())
            return
        _write_json(path, {**summary, 'exit_code': exit_codes.NOT_MEASURED, 'aborted': abort.to_record()})
    except Exception:  # noqa: BLE001 — a report about the run must never BE the run's failure
        logger.warning('could not record the credit stop in %s', path, exc_info=True)


def _publish_credit_stop_before_any_scenario(cfg: _RunConfig, refusal: CreditRefused, settings: Settings) -> int:
    """The credit stop of a run the backend refused before any scenario began: the sentence, the summary, exit 20."""
    block = refusal.credit
    abort = _CreditAbort(
        code=block.code,
        scenario_id=None,
        remedy=credit_sentence(block.code, resets_at=block.resets_at, billing_url=_billing_url(settings)),
        resets_at=block.resets_at,
    )
    _write_run_summary(cfg.out, exit_code=exit_codes.NOT_MEASURED, scenarios={}, graded=set(), cfg=cfg, aborted=abort.to_record())
    _print_measured_stop(abort)
    return exit_codes.NOT_MEASURED


def _reraise_outage(exc: BaseException) -> None:
    """Re-raise *exc* when its request waited out the transient budget (it carries the unavailable block), else return.

    For a read that is best-effort otherwise: the caller ends the run as the unavailable stop, and every other failure keeps
    the caller's own handling.
    """
    if unavailable_block_of(exc) is not None:
        raise exc


def _stop_for_outage_before_any_scenario(cfg: _RunConfig, failure: BaseException) -> int:
    """The unavailable stop of a run that owns ``--out`` and began no scenario because the backend could not serve a read: the
    summary, the text once, exit 20. Re-raises *failure* when it carries no unavailable block (:func:`unavailable_block_of`)."""
    block = unavailable_block_of(failure)
    if block is None:
        raise failure
    abort = _UnavailableAbort(scenario_id=None, remedy=UNAVAILABLE_TEXT, cause=block.cause, code=block.code)
    _write_run_summary(cfg.out, exit_code=exit_codes.NOT_MEASURED, scenarios={}, graded=set(), cfg=cfg, aborted=abort.to_record())
    _print_measured_stop(abort, began_scenarios=False)
    return exit_codes.NOT_MEASURED


def _print_measured_stop(abort: _MeasuredStop, *, began_scenarios: bool = True) -> None:
    """The terminal side of a credit, capacity or unavailable stop: its sentence, once, on stderr.

    A stop met before any scenario began says so, instead of promising results that do not exist.
    """
    kept = 'the scenarios that finished keep their results and the others were not run' if began_scenarios else 'no scenario ran'
    print(
        f'ERROR: the run was stopped ({abort.stopped_by}); {kept}.\n  {abort.remedy}\n  Exit {exit_codes.NOT_MEASURED}: not measured.',
        file=sys.stderr,
    )


#: The ``--trials N`` invocation's SETUP GATE: at most ``--concurrency`` scenario attempts across ALL trials
#: may be in setup (``POST /threads`` through the child's init: configure, session, reset, seed) at once.
#: N trials run N x C agent processes, and N x C setups started on one box in the same seconds can starve
#: it into answering 503. The gate keeps the setup load at what a
#: single run already puts there while the peak of agent processes stays N x C. ``None`` outside a trials
#: run: a single run never has more than C scenarios in flight, so it never needs one.
_SETUP_GATE: ContextVar[asyncio.Semaphore | None] = ContextVar('plurai_setup_gate', default=None)


class _SetupSlot:
    """One scenario attempt's hold on :data:`_SETUP_GATE`, given back ONCE: when the attempt's setup is
    done (``release``, handed to the orchestrator as its ``on_setup_done``), or when the attempt exits
    for any reason before that — a failed create or init must never keep its slot."""

    def __init__(self, gate: asyncio.Semaphore | None) -> None:
        self._gate = gate
        self._held = False

    async def __aenter__(self) -> _SetupSlot:
        if self._gate is not None:
            await self._gate.acquire()
            self._held = True
        return self

    def release(self) -> None:
        if self._held and self._gate is not None:
            self._held = False
            self._gate.release()

    async def __aexit__(self, *_: object) -> None:
        self.release()


#: The gathered strategy's EXCLUSIVE RESOURCES (:class:`_ResourceLeases`, built from ``cfg.resource_leases``): one
#: lease pool per resource only so many agent processes may hold at once — a bounded world's copies and each
#: ``harness.single_instance`` resource. Set ONCE before the gather (:func:`_run_gathered_async`), so every trial and
#: every scenario task it spawns leases from the same pools — the :data:`_SETUP_GATE` pattern. ``None`` when the
#: bench declares none, and always on the experiment strategy: its vendor threads see no ContextVar of this loop,
#: and it keeps the concurrency clamp instead (:func:`_one_at_a_time_for_world`).
_RESOURCE_LEASES: ContextVar[_ResourceLeases | None] = ContextVar('plurai_resource_leases', default=None)


@dataclass(frozen=True, slots=True)
class _LeaseSpec:
    """One exclusive resource a gathered run leases: its run-log ``label``, how many agent processes may hold it at
    once (``copies``), and whether the holder is told WHICH copy it holds (``PLURAI_WORLD_SLOT``, a world of 2+ copies)."""

    label: str
    copies: int
    names_slot: bool = False


class _HeldLeases:
    """One attempt's copies, given back ONCE: when its agent process closed (:meth:`release`, handed to the
    orchestrator as ``on_child_closed``, before ``/finalize``), or when the attempt exits for any reason first."""

    def __init__(self) -> None:
        self._held: list[tuple[_LeaseSpec, asyncio.Queue[int], int]] = []

    def add(self, spec: _LeaseSpec, free: asyncio.Queue[int], copy: int) -> None:
        self._held.append((spec, free, copy))

    @property
    def world_slot(self) -> int | None:
        """The world copy held, for ``PLURAI_WORLD_SLOT``; ``None`` when no held resource names its copies."""
        return next((copy for spec, _, copy in self._held if spec.names_slot), None)

    def release(self) -> None:
        held, self._held = self._held, []
        for _, free, copy in reversed(held):
            free.put_nowait(copy)


class _ResourceLeases:
    """The run's exclusive resources: per :class:`_LeaseSpec`, a queue of its free copy indices (``free``, aligned with
    ``specs``). An attempt takes one copy of each, waiting while none is free, in ONE fixed order so two attempts never
    each hold what the other waits for. An attempt cancelled in its wait takes nothing (``asyncio.Queue.get``)."""

    def __init__(self, specs: Sequence[_LeaseSpec]) -> None:
        self.specs = tuple(specs)
        self.free: tuple[asyncio.Queue[int], ...] = tuple(_free_copies(spec.copies) for spec in self.specs)

    def busy(self) -> _LeaseSpec | None:
        """The first resource with no copy free right now, else ``None``."""
        return next((spec for spec, free in zip(self.specs, self.free, strict=True) if free.empty()), None)

    async def acquire(self) -> _HeldLeases:
        held = _HeldLeases()
        try:
            for spec, free in zip(self.specs, self.free, strict=True):
                held.add(spec, free, await free.get())
        except BaseException:
            # Cancelled (a run abort) while waiting on a later resource: the earlier copies go back.
            held.release()
            raise
        return held


def _free_copies(copies: int) -> asyncio.Queue[int]:
    """A queue holding copy indices ``0..copies-1``."""
    free: asyncio.Queue[int] = asyncio.Queue()
    for copy in range(copies):
        free.put_nowait(copy)
    return free


class _RowLeaseClock:
    """One ROW's waits for the run's leases (:data:`_ROW_LEASE_CLOCK`), across every attempt it makes.

    A row that finds a copy taken says so ONCE (``[wait]``), and its ``[start]`` line and heartbeat begin only when it
    first holds its copies (``begin``): a row waiting for a lease has no agent process, so it is not running. ``waited``
    is every second its attempts spent waiting, which the row's recorded duration leaves out."""

    def __init__(self, scenario_id: str, begin: Callable[[], None]) -> None:
        self.scenario_id = scenario_id
        self.waited = 0.0
        self._begin: Callable[[], None] | None = begin
        self._said_wait = False

    async def acquire(self, leases: _ResourceLeases) -> _HeldLeases:
        busy = leases.busy()
        if busy is not None and not self._said_wait and self._begin is not None:
            self._said_wait = True
            try:
                print(f'[{_ts()}] [wait] {self.scenario_id}: waiting for the {busy.label}')
            except Exception:  # noqa: BLE001 — the announcement is not the run
                logger.debug('could not announce the wait of %s', self.scenario_id, exc_info=True)
        started = time.monotonic()
        try:
            held = await leases.acquire()
        finally:
            self.waited += time.monotonic() - started
        begin, self._begin = self._begin, None
        if begin is not None:
            begin()
        return held


#: The row's :class:`_RowLeaseClock`, set by :func:`_run_one` for the attempts it drives when the run leases resources.
_ROW_LEASE_CLOCK: ContextVar[_RowLeaseClock | None] = ContextVar('plurai_row_lease_clock', default=None)


def _launch_order(count: int, trial: _TrialIdentity | None) -> list[int]:
    """The order a run LAUNCHES its ``count`` scenarios in: trial ``k`` of ``n`` starts at offset
    ``(k - 1) * count // n`` and wraps, so the trials' first waves hold different scenarios and one
    burst against the backend cannot take every trial of the same scenario. A single run is in list order.
    """
    if trial is None or trial.count <= 1 or count <= 1:
        return list(range(count))
    offset = (trial.index - 1) * count // trial.count
    return [(offset + step) % count for step in range(count)]


def _note_host_effects(pairs: Iterable[tuple[str, ThreadResult | None]]) -> None:
    """Remember each row's reported host-effects record (the last turn that carried one), with
    where its successful writes landed (:func:`plurai.host_effects.observed_writes`) when the
    built-in tools were watched: a tool channel was read and some turn carried the trail confine
    recorded (:func:`plurai.host_effects.trail_recorded`; the only channel their writes reach — mock
    events alone never show one, and a returned list nothing appended to is not a recording).

    ``writes_observed['cli_own']`` is folded in separately: the CLI's own session-transcript /
    ``tool-results/`` writes are never a tool event, so :func:`observed_writes` cannot see them —
    :func:`plurai.host_effects.cli_own_write_landed` checks for them directly, in the OPERATOR's own
    config folder (a scenario's own one is not a landing), by the one channel a scenario's ``cwd``
    is knowable through.

    ``reads_watched``: the record's ``reads_observed`` judged at least every read-channel call the
    row's tool events hold (:data:`plurai.host_effects.READ_CHANNEL_TOOLS`) — a harness that reported
    a copy of the record taken before its calls, or none of it, left calls unjudged."""
    for scenario_id, result in pairs:
        trail_carried = False
        for turn in (result.transcript_turns if result is not None else None) or ():
            state = turn.get('agent_state') if isinstance(turn, dict) else None
            record = state.get(HOST_EFFECTS_KEY) if isinstance(state, dict) else None
            if isinstance(record, dict):
                _run_ledgers().host_effects[scenario_id] = record
            trail_carried = trail_carried or trail_recorded(state)
        record = _run_ledgers().host_effects.get(scenario_id)
        events = result.tool_events.get('events') if result is not None and isinstance(result.tool_events, dict) else None
        if record is not None and isinstance(events, list) and trail_carried:
            observed = observed_writes(record, events)
            cwd = record.get('cwd')
            observed['cli_own'] = int(bool(cwd) and cli_own_write_landed(cwd))
            reads = record.get('reads_observed')
            # a call's decision row names the call's tool, and is no call the observer had to judge
            expected = sum(1 for e in events if isinstance(e, dict) and e.get('type') != TOOL_DECISION_EVENT and e.get('tool') in READ_CHANNEL_TOOLS)
            watched = isinstance(reads, dict) and int(reads.get('calls') or 0) >= expected
            _run_ledgers().host_effects[scenario_id] = {**record, 'writes_observed': observed, 'reads_watched': watched}


def _note_tool_channel(pairs: Iterable[tuple[str, ThreadResult | None]], simulate_path: Path) -> None:
    """Remember each row of a bench that declares ``agent.native_tools`` where no turn declared the
    trail lane (:func:`plurai.host_effects.trail_recorded`): those tools ran where nothing measured
    them, whatever the mock lane saw, so the row is not measured (:func:`_measurement_unchecked_because`),
    never clean. A bench declaring none is untouched."""
    rows = [
        (sid, result)
        for sid, result in pairs
        if result is not None and not any(isinstance(turn, dict) and trail_recorded(turn.get('agent_state')) for turn in result.transcript_turns or ())
    ]
    if rows and (native := _declared_native_tools(simulate_path)):
        _run_ledgers().tool_channel_unread.update({sid: list(native) for sid, _result in rows})


def _note_mock_unanswered(pairs: Iterable[tuple[str, ThreadResult | None]]) -> None:
    """Remember each row whose agent was served a stub for a tool call the mock never answered:
    ``{calls, restart_shaped, served_schema_stub}``. A measurement breach the backend cannot see —
    it graded a dialog in which some tool answers were the runner's stubs — so the row folds into
    ``measurement_clean`` (:func:`_measurement_unclean_because`). The typed per-call record is the
    run's ``mock.tool_call_unanswered`` finding (:func:`_record_unanswered_calls`).

    A row whose child CLAIMED NOTHING (``unanswered_mock_calls is None``) is never read as every call
    answered: it is filed as unclaimed, so ``measurement_unchecked_because`` names ``mock_unanswered``."""
    for scenario_id, result in pairs:
        if result is None:
            continue
        if result.unanswered_mock_calls is None:
            _run_ledgers().mock_unanswered_unclaimed[scenario_id] = True
            continue
        calls = result.unanswered_mock_calls
        if calls:
            _run_ledgers().mock_unanswered[scenario_id] = {
                'calls': len(calls),
                'restart_shaped': restart_shaped_calls(calls),
                'served_schema_stub': sum(1 for call in calls if isinstance(call, dict) and call.get('served') == 'schema_stub'),
            }


def _note_ipc_trimmed(pairs: Iterable[tuple[str, ThreadResult | None]]) -> None:
    """Remember each row whose agent replies the child trimmed to fit one IPC line, with how many turns. A
    measurement breach only this side sees — the backend graded those turns without their message chain — so the
    row folds into ``measurement_clean`` (:func:`_measurement_unclean_because`); the turns are the run's
    ``ipc.reply_trimmed`` finding (:func:`_record_landing_findings`)."""
    for scenario_id, result in pairs:
        if result is not None and result.ipc_trimmed_turns:
            _run_ledgers().ipc_trimmed[scenario_id] = len(result.ipc_trimmed_turns)


def _record_unanswered_calls(scenario_id: str, calls: list[dict[str, Any]]) -> None:
    """THE ``mock.tool_call_unanswered`` finding for one scenario: which calls went unanswered, what the
    agent was served for each, and what would get them answered. Advisory and non-holding: the
    scenario is graded and the exit code does not move. Evidence is the typed per-call record only
    (no message text), so the fingerprint is stable across runs over the same dialog."""
    # Read field by field: a key a newer child adds is not evidence this gate declared. ``resets_at`` and ``reset_source``
    # are the credit and capacity blocks', and unavailable the unavailable block's: each ends the scenario before this finding is
    # written (credit and capacity stop the run too).
    declared = [name for name in UnansweredCall._fields if name not in ('resets_at', 'reset_source', 'unavailable')]
    records = [UnansweredCall(**{name: call.get(name) for name in declared}) for call in calls if isinstance(call, dict)]
    remedies = list(dict.fromkeys(unanswered_call_remedy(call) for call in records))
    current_gate_authority().emit(
        make_finding(
            MOCK_TOOL_CALL_UNANSWERED,
            subject=scenario_id,
            evidence={
                'calls': [{name: getattr(call, name) for name in declared} for call in records],
                'calls_unanswered': len(records),
                'restart_shaped': sum(1 for call in records if call.restart_shaped),
                'served_schema_stub': sum(1 for call in records if call.served == 'schema_stub'),
            },
            observed={},
            remedy='; '.join(remedies),
            draft_ref=None,
        )
    )


def _with_landing_evidence(result: ThreadResult, progress: ScenarioProgress | None) -> ThreadResult:
    """``result`` with its landed recursion stamped with ``pre_crash_tool_activity`` (from this attempt's
    own tool events, :func:`_pre_crash_tool_activity`) — the evidence that keeps an empty mock's loop
    from reading as the agent's. Every other result is returned as it is."""
    failure = result.turn_failure
    if result.ended_early != AGENT_RECURSION_REASON or not isinstance(failure, dict) or 'pre_crash_tool_activity' in failure:
        return result
    return dataclasses_replace(result, turn_failure={**failure, 'pre_crash_tool_activity': _pre_crash_tool_activity(progress)})


def _record_landing_findings(scenario_id: str, result: ThreadResult) -> None:
    """The findings a graded dialog carries about how it LANDED: a turn failure it was graded to
    (``dialog.turn_failure_landed``; a provider or per-turn-wall cut is also ``dialog.provider_infra_invalid``
    / ``dialog.turn_budget_invalid``, which holds that trial cell), a snapshot channel left unread
    (``dialog.world_capture_failed``), world values carried in their JSON form (``world.values_coerced``),
    and interception files the agent changed (``mock.interception_changed``). None changes the exit code."""
    authority = current_gate_authority()
    failure = result.turn_failure
    if isinstance(failure, dict):
        authority.emit(
            make_finding(
                DIALOG_TURN_FAILURE_LANDED,
                subject=scenario_id,
                evidence={
                    'reason': result.ended_early if not failure.get('closing') else None,
                    'kind': failure.get('kind'),
                    'turn_idx': failure.get('turn_idx'),
                    'closing': bool(failure.get('closing')),
                    'child_exc_class': failure.get('child_exc_class'),
                },
                observed={'error': failure.get('error'), 'pre_crash_tool_activity': failure.get('pre_crash_tool_activity')},
                remedy=_reason_route(result.ended_early)[1]
                if result.ended_early and not failure.get('closing')
                else 'the closing reply was not delivered; read turn_failure',
                draft_ref=None,
            )
        )
        if (cut := harness_cut_finding(result.ended_early, scenario_id, failure)) is not None:
            authority.emit(cut)
    if result.capture_failed:
        authority.emit(
            make_finding(
                DIALOG_WORLD_CAPTURE_FAILED,
                subject=scenario_id,
                evidence={'channels': sorted(result.capture_failed)},
                observed=dict(result.capture_failed),
                remedy='the mock could not be read at finalize even after it proved ready: check that the mock is up and its session alive, then re-run',
                draft_ref=None,
            )
        )
    if result.world_values_coerced:
        authority.emit(
            make_finding(
                WORLD_VALUES_COERCED,
                subject=scenario_id,
                evidence={'paths': dict(sorted(result.world_values_coerced.items()))},
                observed={},
                remedy='return JSON-native values from world_snapshot() when a policy must compare them in their original type',
                draft_ref=None,
            )
        )
    if result.interception_changed:
        print(
            f'[world-state] {scenario_id}: the agent changed {", ".join(result.interception_changed)}, which routes its vendor calls to '
            'the mock; calls after the change may have reached the real service, and the world holds none of their effects.',
            file=sys.stderr,
        )
        authority.emit(
            make_finding(
                MOCK_INTERCEPTION_CHANGED,
                subject=scenario_id,
                evidence={'files': list(result.interception_changed)},
                observed={},
                remedy="read this row's transcript before trusting its world checks: vendor calls after the change may have reached the real "
                'service, and the world holds none of their effects',
                draft_ref=None,
            )
        )
    if result.ipc_trimmed_turns:
        authority.emit(
            make_finding(
                IPC_REPLY_TRIMMED,
                subject=scenario_id,
                evidence={'turns': sorted({int(turn['turn_idx']) for turn in result.ipc_trimmed_turns if isinstance(turn.get('turn_idx'), int)})},
                observed={},
                remedy='some replies did not fit one IPC line and crossed without their message chain, so judges reading those turns saw '
                'less than the agent did: return a smaller agent_state / message chain from the harness for those turns',
                draft_ref=None,
            )
        )


def _record_ipc_overrun(scenario_id: str, exc: HarnessIpcOverrun) -> None:
    """``ipc.line_overrun``: an envelope did not fit one IPC line. Invariant — the attempt has no reply to
    grade — and the row already fails through its ``error.json``; the finding types why."""
    current_gate_authority().emit(
        make_finding(
            IPC_LINE_OVERRUN,
            subject=scenario_id,
            evidence={'envelope_type': exc.envelope_type, 'direction': exc.direction, 'limit': exc.limit},
            observed={'error': str(exc)},
            remedy='keep bulk out of the reply and the world snapshot (or the seeded world): one envelope must fit the IPC line limit',
            draft_ref=None,
        )
    )


def _host_effects_summary() -> dict[str, Any]:
    """Every consented host-effect tool and every deny (with each distinct way it was applied).

    ``{}`` when no harness reported: "nothing was reported" must not read as "confined".
    """
    rows = _run_ledgers().host_effects
    if not rows:
        return {}
    granted: set[str] = set()
    denied: dict[str, set[str]] = {}
    for record in rows.values():
        granted.update(str(tool) for tool in record.get('granted') or ())
        for tool, how in (record.get('denied') or {}).items() if isinstance(record.get('denied'), dict) else ():
            denied.setdefault(str(tool), set()).add(str(how))
    # Reads are ``scoped`` only where every read-channel call was judged (``reads_watched``) and every
    # surface that can read was confined: a record that names neither is unobserved, and its
    # declared ``reads`` is not an observation.
    reads_complete = all(r.get('reads_watched') is True and _read_surfaces_confined(r) for r in rows.values())
    # Likewise ``mcp``: a record that does not say ``declared_only`` is read as unconfined.
    declared_only = all(record.get('mcp') == 'declared_only' for record in rows.values())
    observed = [record.get('writes_observed') for record in rows.values()]
    # a consented (live) Bash writes where no tool event says, and a record naming no shell home cannot show where
    # its shell wrote: neither row is observed. Nor is a
    # row whose writes_observed never checked 'cli_own' — an unread channel is not a clean one, the
    # same idiom as the tool-event channel above it. Nor is a row whose operator let a program out of
    # the Bash sandbox (``unsandboxed_grant``): a command the CLI runs outside the sandbox writes where
    # no tool event says either. A record with no grant key comes from a confine that let nothing out.
    complete = all(
        isinstance(r.get('writes_observed'), dict)
        and 'cli_own' in r['writes_observed']
        and not _grant_names(r)
        and 'Bash' not in (r.get('granted') or ())
        and isinstance(r.get('shell_home'), str)
        for r in rows.values()
    )
    return {
        'granted': sorted(granted),
        'denied': {tool: sorted(hows) for tool, hows in sorted(denied.items())},
        'scenarios': sorted(rows),
        **_reads_landed((r.get('reads_observed') for r in rows.values()), complete=reads_complete),
        'reads_not_confined': sorted({name for r in rows.values() for name in _reads_not_confined(r)}),
        'mcp': 'declared_only' if declared_only else 'unconfined',
        # how each scenario's CLI signed in, by route name only (``plurai.host_effects.CLI_CREDENTIAL_ENV``)
        'auth': sorted({str(r['auth']) for r in rows.values() if r.get('auth')}),
        # where each scenario's CLI kept its config and transcripts (``plurai.host_effects.CLI_CONFIG_*``)
        'cli_config': sorted({str(r['cli_config']) for r in rows.values() if r.get('cli_config')}),
        **_grant_rollup(rows.values()),
        **_writes_landed(observed, complete=complete),
    }


#: ``host_effects.disclosure``: no artifact that ships names the operator's home out of scope, or one does.
DISCLOSURE_CLEAN = 'clean'
DISCLOSURE_HOME_NAMED = 'home_named'
_DISCLOSURE_KEYS = ('disclosure', 'disclosed_home_paths', 'disclosure_scenarios')


def _home_disclosure(out: Path, scenario_ids: Iterable[str], *, repo: Path) -> dict[str, Any]:
    """What this run's shipped artifacts disclose of the operator's home, for every agent shape:
    :meth:`plurai.host_effects.HostReadScope.home_named` over each scenario's
    :data:`SCENARIO_ARTIFACT_FILES` (and its ``stability_retry/`` copies), by the scope
    :func:`plurai.host_effects.scrub_parent_record` reads (``repo`` allowed). Its own axis beside
    ``reads`` and ``writes``: a path named is not a path read. A file that is not JSON is counted as text."""
    scope = host_read_scope(run_dir=None, repo=repo, harness=None)
    named: dict[str, int] = {}
    for sid in scenario_ids:
        for folder in (out / sid, out / sid / STABILITY_RETRY_DIRNAME):
            for name in SCENARIO_ARTIFACT_FILES:
                try:
                    text = (folder / name).read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                try:
                    obj: Any = json.loads(text)
                except ValueError:
                    obj = text
                if n := scope.home_named(obj):
                    named[sid] = named.get(sid, 0) + n
    return {
        'disclosure': DISCLOSURE_HOME_NAMED if named else DISCLOSURE_CLEAN,
        'disclosed_home_paths': sum(named.values()),
        'disclosure_scenarios': sorted(named),
    }


def _union_disclosure(blocks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """:func:`_home_disclosure` over N trials; ``{}`` when no trial carried one (a summary without the block)."""
    carried = [block for block in blocks if block.get('disclosure') in (DISCLOSURE_CLEAN, DISCLOSURE_HOME_NAMED)]
    if not carried:
        return {}
    scenarios = {str(sid) for block in carried for sid in block.get('disclosure_scenarios') or ()}
    return {
        'disclosure': DISCLOSURE_HOME_NAMED if any(block['disclosure'] == DISCLOSURE_HOME_NAMED for block in carried) else DISCLOSURE_CLEAN,
        'disclosed_home_paths': sum(int(block.get('disclosed_home_paths') or 0) for block in carried),
        'disclosure_scenarios': sorted(scenarios),
    }


def _reads_not_confined(record: Mapping[str, Any]) -> list[str]:
    """The read surfaces a record (or a trials block) names as not confined."""
    surfaces = record.get('read_surfaces')
    names = record.get('reads_not_confined')
    if isinstance(surfaces, dict):
        return [str(name) for name, how in surfaces.items() if how != READ_CONFINED]
    return [str(name) for name in names] if isinstance(names, list) else []


def _read_surfaces_confined(record: Mapping[str, Any]) -> bool:
    """Whether a record names its read surfaces, every one of them confined."""
    surfaces = record.get('read_surfaces')
    return isinstance(surfaces, dict) and bool(surfaces) and not _reads_not_confined(record)


def _reads_landed(observed: Iterable[Any], *, complete: bool) -> dict[str, Any]:
    """``reads_observed`` summed, and ``reads``:

    * ``escaped``: a call got content back from a path out of scope (``outside``);
    * ``scoped``: every read surface was confined and every read-channel call judged (``complete``),
      and none named an out-of-scope path it left undecided (``unseen``) — the scope was enforced and
      nothing observed contradicts it. It does not claim every read was seen: a path a command builds
      at run time (``$VAR/x``, a glob) is not named, so only the sandbox holds it;
    * ``unobserved``: anything less — a declared deny is not an observed one."""
    counts = [row for row in observed if isinstance(row, dict)]
    total = {key: sum(int(row.get(key) or 0) for row in counts) for key in READ_LANDINGS}
    reads = 'escaped' if total['outside'] else 'scoped' if complete and not total['unseen'] else 'unobserved'
    return {'reads_observed': total, 'reads': reads}


def _grant_names(block: Mapping[str, Any]) -> list[str]:
    """The programs a record's (or a summary's) ``unsandboxed_grant`` let out of the Bash sandbox."""
    grant = block.get('unsandboxed_grant')
    return [str(name) for name in grant.get('names') or ()] if isinstance(grant, dict) else []


def _grant_rollup(blocks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The Bash-sandbox grant over records or trial summaries: every program let out
    (``unsandboxed_grant.names``) and every declared program that ran inside the sandbox without a
    grant (``sandboxed_executables``), each unioned. The read surfaces a grant leaves unconfined are
    ``reads_not_confined`` (:func:`_reads_not_confined`, from each record's ``read_surfaces``)."""
    seen = list(blocks)
    return {
        'unsandboxed_grant': {'names': sorted({name for block in seen for name in _grant_names(block)})},
        'sandboxed_executables': sorted({str(name) for block in seen for name in block.get('sandboxed_executables') or ()}),
    }


def _writes_landed(observed: Iterable[Any], *, complete: bool) -> dict[str, Any]:
    """``writes_observed`` summed, and ``writes``: ``escaped`` when one landed outside the scenario's
    folders OR the CLI's own project dir gained one (``cli_own`` — the CLI writing its session
    transcript into the operator's real home is exactly as escaped as the agent doing it),
    ``confined`` only when every row was observed (``complete``) — an unread tool channel is not a clean one."""
    counts = [row for row in observed if isinstance(row, dict)]
    total = {key: sum(int(row.get(key) or 0) for row in counts) for key in WRITE_LANDINGS}
    if total['outside'] or total['cli_own']:
        writes = 'escaped'
    elif complete:
        writes = 'confined'
    else:
        writes = 'unobserved'
    return {'writes_observed': total, 'writes': writes}


def _reset_run_ledgers() -> None:
    """Empty the ACCUMULATING module ledgers at run start, before a scenario can add to them.

    Their three siblings — ``_LAST_EGRESS_RECORD``, ``_LAST_SCAN_DRIFT_RECORD``,
    ``_LAST_SOLVABILITY_RECORD`` — each hold ONE verdict and clear inside their own recorder,
    which every run reaches unconditionally. These two ACCUMULATE per scenario, so they have no
    such moment: without this reset a second run in the same process would inherit the first's
    contents, and ``_aggregate_backend_build`` would report ``mixed`` — a deployment-drift alarm
    manufactured by two runs against the same backend — beside another run's
    ``untrustworthy_scenarios``.

    Called from BOTH strategies, beside :func:`_clear_stale_run_summary` and for the same reason:
    a previous run's facts must not survive into this one's summary.

    The orchestrator's two warning THROTTLES are the same class one package over, so they are
    emptied from the same moment: their own docstrings promise "one line ... for the whole run",
    and a second run in one interpreter must still report a condition it hits.
    """
    global _RUN_ABORT  # noqa: PLW0603 — the run-wide abort is reset with the ledgers
    ledgers = _run_ledgers()
    ledgers.backend_stamps.clear()
    ledgers.untrustworthy.clear()
    ledgers.host_effects.clear()
    ledgers.harness_fault.clear()
    ledgers.not_run.clear()
    ledgers.tool_channel_unread.clear()
    ledgers.mock_unanswered.clear()
    ledgers.mock_unanswered_unclaimed.clear()
    ledgers.ipc_trimmed.clear()
    ledgers.credit_waits.clear()
    _CREDIT_WAITS.reset()
    with _RUN_ABORT_LOCK:
        _RUN_ABORT = None
    reset_warning_ledgers()


def _record_untrustworthy(scenario: str, result: ThreadResult) -> str | None:
    """This row's untrustworthy-world reason, remembered for the summary and returned for the row.

    SAID on the way past, not just filed. :func:`_policy_cells` deliberately keeps the cells a
    distrusted world still produced, so ``scored: false`` is not the signal that something
    was wrong with them — and without a line here the only trace of it would be a key in two JSON
    files. The vacuous-green hazard is real (every ``entities_of_type(...)`` resolves to ``[]`` on
    a zero-entity world), so the caveat has to reach the operator who reads the number.
    """
    reason = _untrustworthy_world(result)
    if reason:
        _run_ledgers().untrustworthy[scenario] = reason
        print(
            f'[world-state] UNTRUSTWORTHY: {scenario} was scored over a world the backend does not vouch for '
            f'({reason}). Its cells are kept — they landed — but a check that reads the world resolves against '
            f'nothing there, so a pass on this row may be vacuous. See untrustworthy_scenarios in {RUN_SUMMARY_FILENAME}.',
            file=sys.stderr,
        )
    return reason


def _record_backend_stamp(finalize: Any) -> dict[str, Any] | None:
    """This scenario's backend stamp, remembered for the summary and returned for the transcript.

    ``None`` — never an omitted key — when the backend did not send one. "This run was measured
    somewhere unknown" and "this writer does not record where" are different claims, and only a
    written ``null`` tells them apart.
    """
    stamp = getattr(finalize, 'backend', None)
    if stamp is None:
        return None
    payload = stamp.model_dump() if hasattr(stamp, 'model_dump') else dict(stamp)
    stamps = _run_ledgers().backend_stamps
    if payload not in stamps:
        stamps.append(payload)
    return payload


def _aggregate_backend_build() -> dict[str, Any] | None:
    """The backend that measured this RUN, or ``None`` when no row recorded one.

    Distinct stamps ride under ``mixed`` rather than being collapsed to the first: two of them
    means the backend restarted or was redeployed between scenarios, and a run scored half on
    each is exactly the thing this field makes visible.
    """
    stamps = _run_ledgers().backend_stamps
    if not stamps:
        return None
    head = stamps[0]
    return head if len(stamps) == 1 else {**head, 'mixed': list(stamps)}


def _stamped_solvability(metadata: Any) -> str:
    """One scenario's stamped solvability state, read off its fetched metadata.

    A MIRROR of the backend's own solvability parser, deliberately including its
    tolerance: an absent, malformed or unrecognised field is ``untested`` — a scenario without the
    stamp HAS not been checked — and it never reads as ``solvable``, which is
    the only direction that would matter.

    IDENTITY checks, not a lookup table: a table reads ``solvable: 1`` as ``solvable``
    (``1 == True`` in Python) — a truthy int deciding a quarantine question — and raises
    ``TypeError`` on an unhashable value, on a path where upstream is explicitly total because a
    malformed field must not take down a run.
    """
    value = metadata.get('solvable') if isinstance(metadata, dict) else None
    if value is True:
        return 'solvable'
    if value is False:
        return 'unsolvable'
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {'true', 'solvable'}:
            return 'solvable'
        if token in {'false', 'unsolvable'}:
            return 'unsolvable'
    return 'untested'


async def _preflight_solvability(cfg: _RunConfig, settings: Settings) -> dict[str, Mapping[str, Any]] | None:
    """Read each scenario's solvability stamp from the backend, AT RUN START.

    The stamp rides ``scenario.yaml``'s ``scenario:`` block and reaches a client only through
    ``GET /scenario-sets/{id}``, which flattens that block into each entry's metadata. Fetching
    it HERE — rather than trusting a list resolved earlier, or a local copy the plugin
    materialised before adoption — is what makes a stamp written by a post-qualification
    adoption travel to the run that ships numbers.

    REPORT ONLY, and deliberately: the backend refuses a quarantined scenario at ``POST
    /threads``, which is the one chokepoint every scored run goes through, and refusing the whole
    run here because one scenario of seven is quarantined would throw away six measurements this
    run can still take. What this adds is that an operator learns it BEFORE the subprocesses
    spend, and that ``run_summary.json`` carries the census instead of the fact being
    reconstructable only from per-scenario ``error.json`` files.

    A fetch that fails records ``read: false``. Never an empty census: "no scenario is
    quarantined" and "nobody looked" are different runs.

    One extra key-free GET rather than threading the gather strategy's own fetch through: the two
    strategies must answer this identically, and the experiment strategy with explicit
    ``--scenario-id`` makes no set fetch at all.

    Returns the flattened ``scenario:`` metadata of the scenarios THIS run will attempt, keyed by
    scenario id, for :func:`_resolve_max_turns`; ``None`` when the set could not be read.
    """
    _LAST_SOLVABILITY_RECORD.clear()
    _LAST_SCENARIO_SET_RECORD.clear()
    _LAST_SCENARIO_LIKELIHOODS.clear()
    try:
        client = build_default_client(settings)
        try:
            fetched = await client.fetch_scenario_set(cfg.scenario_set_id)
        finally:
            await client.aclose()
        # The stamp READ is inside the same guard as the fetch: a census documented as
        # report-only may not decide a run's fate over one malformed entry.
        stamps = {entry.folder_name: _stamped_solvability(entry.metadata) for entry in (fetched.scenarios or []) if entry.folder_name}
        metadata = {entry.folder_name: entry.metadata for entry in (fetched.scenarios or []) if entry.folder_name and isinstance(entry.metadata, Mapping)}
    except Exception as exc:  # noqa: BLE001 — a census must never be able to refuse a run
        _LAST_SOLVABILITY_RECORD.update({'read': False, 'detail': _failure_text(exc)})
        print(
            f'[solvability] NOT READ: the scenario set could not be read ({_failure_text(exc)}), so this run does '
            'not know which of its scenarios the suite has quarantined.',
            file=sys.stderr,
        )
        return None
    _LAST_SCENARIO_SET_RECORD.update(_scenario_set_record(fetched))
    _LAST_SCENARIO_LIKELIHOODS.update({folder: meta.get('path_probability') for folder, meta in metadata.items()})
    # The population is what THIS run will attempt, not the whole set: a --scenario-id subset that
    # excludes the quarantined scenario is not a run with a quarantine problem.
    wanted = list(cfg.scenario_ids) if cfg.scenario_ids else sorted(stamps)
    census: dict[str, int] = {}
    quarantined: list[str] = []
    for scenario_id in wanted:
        state = stamps.get(scenario_id, 'not_in_set')
        census[state] = census.get(state, 0) + 1
        if state == 'unsolvable':
            quarantined.append(scenario_id)
    # Only a DECIDED stamp is an examination. `untested` is the check saying it did not run for
    # that scenario, and `not_in_set` is the set not carrying it at all — neither can contribute
    # to an all-clear. This is not a corner: generation never writes `solvable`, so an
    # all-`untested` census is the usual state, and a clean bill over it would vouch for a check
    # nobody ran.
    examined = sum(count for state, count in census.items() if state in {'solvable', 'unsolvable'})
    _LAST_SOLVABILITY_RECORD.update({'read': True, 'examined': examined, 'census': census, 'quarantined_scenarios': sorted(quarantined)})
    if quarantined:
        print(
            f'[solvability] {len(quarantined)} scenario(s) are QUARANTINED and the backend will refuse to open a thread for '
            f'them — their reference path cannot pass their own eval set, so scoring an agent against them would measure the '
            f'suite: {", ".join(sorted(quarantined))}',
            file=sys.stderr,
        )
    elif examined:
        print(f'[solvability] no scenario in this run is quarantined ({_census_summary(census)}).', file=sys.stderr)
    else:
        # "Nothing was examined" must not print as "nothing was found". A set that served no scenario (or served none of the
        # requested ids) supports no all-clear.
        print(
            f'[solvability] NOT EXAMINED: no scenario this run will attempt has a solvability verdict ({_census_summary(census)}). '
            + (
                'To examine them, run the solvability check on this scenario set (`solvability_check_start`, the qualification '
                'step of /simulation-generate-suite).'
                if census.get('untested')
                else 'The scenario set does not carry them: check --scenario-set-id and --scenario-id.'
            ),
            file=sys.stderr,
        )
    return {scenario_id: metadata[scenario_id] for scenario_id in wanted if scenario_id in metadata}


def _census_summary(census: dict[str, int]) -> str:
    return ', '.join(f'{count} {state}' for state, count in sorted(census.items())) or 'no scenario'


#: The turn cap a run takes when the operator named none and its own scenarios cannot size one,
#: and the FLOOR under every derived cap: a derived budget is never smaller than this, so no run
#: gets fewer turns than it would have without the derivation.
DEFAULT_MAX_TURNS = 12

#: Where a run's turn cap came from: the operator's ``--max-turns``, the run's own scenario floors,
#: or :data:`DEFAULT_MAX_TURNS` because the floors could not be read.
MaxTurnsSource = Literal['flag', 'derived', 'fallback']


def _int_field(row: Mapping[str, Any], key: str) -> int | None:
    value = row.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def derive_max_turns(rows: Sequence[Mapping[str, Any]], *, opens: bool) -> int | None:
    """The turn cap a run's OWN scenarios need, or ``None`` when no row says how long it runs.

    Each scenario needs its published minimum (``estimated_min_turns``) plus two turns of slack for
    the conversation to close, plus headroom that grows with its ``complexity_level`` (at least 2;
    a missing level reads as 2). The run takes the longest scenario's need, one more turn when the
    agent opens the conversation (that turn spends the cap without being one of the scenario's),
    and never less than :data:`DEFAULT_MAX_TURNS`. A conversation cut at the cap scores the
    obligations it never reached, so the cap is sized for the longest scenario, not the typical one.

    ``rows`` are the metadata of the scenarios THIS run plays (a ``--scenario-id`` subset or the
    whole set); a row without an integer ``estimated_min_turns`` contributes nothing.
    """
    needs = [minimum + 2 + max(2, _int_field(row, 'complexity_level') or 2) for row in rows if (minimum := _int_field(row, 'estimated_min_turns')) is not None]
    if not needs:
        return None
    return max(DEFAULT_MAX_TURNS, max(needs) + (1 if opens else 0))


def _harness_opens(simulate_path: Path) -> bool:
    """Does the agent open the conversation (``agent.io`` ``may_open`` / ``opens: always``)? The same
    declaration the turn loop dispatches turn 0 on. ``main()`` has already refused a config that
    cannot be read; here an unreadable one reads as "does not open", a one-turn difference."""
    try:
        return (declared_agent_io(simulate_path) or DEFAULT_AGENT_IO).may_open
    except (WorkspaceConfigError, OSError):
        return False


def _resolve_max_turns(cfg: _RunConfig, rows: Mapping[str, Mapping[str, Any]] | None) -> _RunConfig:
    """The run's turn cap, resolved ONCE after the set fetch: the operator's flag, else derived from
    the scenarios this run plays (:func:`derive_max_turns`), else :data:`DEFAULT_MAX_TURNS` with one
    warning. The source travels on the config into ``run_summary.json``."""
    if cfg.max_turns is not None:
        return cfg if cfg.max_turns_source is not None else dataclasses_replace(cfg, max_turns_source='flag')
    derived = derive_max_turns(list(rows.values()), opens=_harness_opens(cfg.agent_module_path)) if rows is not None else None
    if derived is None:
        print(
            f'[turn-budget] WARNING: the scenarios this run plays carry no readable turn floors, so the cap is the default '
            f'{DEFAULT_MAX_TURNS} turns; pass --max-turns to size it for a long set.',
            file=sys.stderr,
        )
        return dataclasses_replace(cfg, max_turns=DEFAULT_MAX_TURNS, max_turns_source='fallback')
    return dataclasses_replace(cfg, max_turns=derived, max_turns_source='derived')


def _publish_eval_set_record(record: dict[str, Any], warning: str | None) -> None:
    """Replace the eval-set record, and send any warning down BOTH channels.

    Two channels because they answer to two different readers: stderr for the operator running the
    command, and ``run_summary.json`` for everything that reads the run afterwards. A warning that
    exists only in a terminal is a warning a re-scored verdict cannot be checked against later.
    One writer so neither producer
    can publish to one channel and forget the other.
    """
    _LAST_EVAL_SET_RECORD.clear()
    _LAST_EVAL_SET_RECORD.update(record)
    if warning is not None:
        _LAST_EVAL_SET_RECORD['warning'] = warning
        print(f'[eval-set] WARNING: {warning}', file=sys.stderr)


def _record_eval_set_state(eval_set_id: str, answer: EvalSetPoliciesResponse) -> None:
    """Record WHICH BYTES answered the pre-flight, and warn when they were a mid-edit draft.

    THREE STATES, and the fourth is ``None``. ``draft`` warns. ``promoted`` is silent; ``unknown``
    prints a NOTE — a backend with its edit-session store wired never reports it, so `unknown` means
    the draft axis is dark for every run on that backend, which is exactly the
    kind of silence this note exists to break. ``state is None`` is a server that does not send the field,
    recorded as ``None`` rather than normalized: "this backend cannot tell" and "this client asked
    a backend that does not know the question" are different facts about different fleets.

    A value from outside the documented set is reported rather than ignored: a newer server
    state must not silently read as "not a draft, carry on".

    Typed, and read by ATTRIBUTE. A ``getattr(answer, 'state', None)`` would turn a renamed or
    dropped model field into the silent claim "this server predates it" — a false statement about
    the deployed fleet, and the one state that prints nothing at all.
    """
    state, session = answer.state, answer.edit_session_id
    warning = None
    if state == _EVAL_SET_DRAFT:
        warning = (
            f'eval set {eval_set_id} is a DRAFT: edit session {session or "(unnamed)"} is OPEN on it, so this run is '
            f"scored by that session's in-progress policies_config.yaml — bytes end_edit_session may still refuse. "
            f"The verdicts are real; they are just not the committed set's. Re-run after the session commits before "
            f"citing them as the suite's result."
        )
    elif state is not None and state not in _EVAL_SET_STATES:
        warning = (
            f'eval set {eval_set_id} reports state {state!r}, which this client does not know '
            f'(it knows {sorted(_EVAL_SET_STATES)}). It is NOT being read as "promoted" — upgrade the runner, or '
            f"check the backend, before treating these verdicts as the committed set's."
        )
    elif state == _EVAL_SET_UNKNOWN:
        print(
            f'[eval-set] NOTE: eval set {eval_set_id} reports state "{_EVAL_SET_UNKNOWN}" — this backend has no '
            f"edit-session store wired, so it cannot tell a committed set from an open session's draft. Not a "
            f'warning about this run; a statement that the draft axis is dark on this backend.',
            file=sys.stderr,
        )
    _publish_eval_set_record({'eval_set_id': eval_set_id, 'state': state, 'edit_session_id': session}, warning)


def _record_eval_set_unverified(eval_set_id: str, why: str) -> None:
    """The pre-flight ASKED and could not be answered — a fourth record, not an empty one.

    `{}` is written when no eval set was attached at all ("nobody asked"). A fetch that 404'd into
    the warn-and-proceed branch, timed out, or came back unparseable is a different run: the
    question was asked and the answer is missing, so the draft axis is dark for a reason that
    belongs beside the verdicts. `state: None` with an `unverified` reason, so a reader can tell it
    from a backend that simply predates the field.
    """
    _publish_eval_set_record(
        {'eval_set_id': eval_set_id, 'state': None, 'edit_session_id': None, 'unverified': why},
        f'the eval set {eval_set_id} could not be verified ({why}), so whether this run was scored by the committed '
        'set or by an open edit session\'s draft is UNKNOWN — not "promoted".',
    )


async def _preflight_eval_set(cfg: _RunConfig, settings: Settings) -> str | None:
    """Refuse a run whose ``--eval-set-id`` the backend has already rejected.

    A no-op without ``--eval-set-id``. Otherwise one cheap authenticated read of
    ``GET /eval-sets/{id}/policies``, returning the operator-facing error string (shaped
    like :func:`_preflight_backend`'s) when the run must not start, else ``None``.

    Posture, deliberately asymmetric:

    * HTTP **400 / 404** ⇒ HARD FAIL. These are the two statuses the endpoint documents,
      and both are definitive statements about THIS eval set: the same policy
      build that ``POST /threads`` performs already rejected it, so every scenario worker
      would die on the identical 400. Re-sending the same bytes cannot change the answer;
      it only spends N agent subprocesses to learn it N times.
    * An **empty policy list** ⇒ HARD FAIL. The endpoint answered, and its answer is that
      this eval set grades nothing: every scenario would finalize over zero policies and
      report ``pass_rate 0.000`` — byte-identical to a run that failed every policy — while
      the summary announced "N/N scenarios succeeded" with no not-scored banner anywhere.
      The fetched list is INSPECTED, not merely checked for a raise, because that one call
      already carries the verdict. Re-sending the same id cannot change the answer, exactly
      as for the 400 above.
    * A read that waited out the transient budget (502 / 503 / 504, or no connection, for about 60
      seconds) ⇒ the error propagates tagged, and the caller ends the run as the unavailable stop.
    * Anything else — another status, any other transport error, a
      response that fails schema validation — ⇒ WARN on stderr and PROCEED. A blip must
      never refuse a run the box could serve; liveness is already
      :func:`_preflight_backend`'s job and this gate must not become a second, weaker one.

    Accepted caveat: a backend so old it lacks the route also answers 404 and is refused
    here. Deliberate — it matches the repo's existing "version skew fails loudly at the
    resolve seams" posture (``client._resolve_server_paths``), and such a backend cannot
    serve this run anyway. A server that OMITS the ``policies`` field is refused by the
    empty-list rule for the same reason: the client logs that contract violation and
    degrades to ``[]``, and a run that cannot establish it has policies must not be allowed
    to report that it graded any.

    A **DRAFT** eval set is the one thing here that WARNS and never refuses. The
    same fetch also answers WHICH BYTES it answered from (``state``), and ``draft`` means
    an edit session is open on the set, so the names — and the rubrics about to grade this run
    — are that session's mid-edit working copy, which ``end_edit_session`` may still refuse.
    Scoring a draft is exactly what the edit loop wants ("did my in-progress fix flip the
    cell?"), so refusing would break the loop; what must never happen is the verdict being
    read as the committed set's. The warning goes to stderr AND into ``run_summary.json``,
    because a run's verdict outlives the terminal it was printed in.
    """
    # Cleared HERE, not only written on success: the record is module-level, so an invocation
    # that attaches no eval set (or whose fetch never answers) must publish an EMPTY record
    # rather than whatever the last one left there. `{}` is the honest "nobody asked".
    _LAST_EVAL_SET_RECORD.clear()
    if not cfg.eval_set_id:
        return None
    client = None
    # ``None`` until the endpoint actually answers — the warn-and-proceed paths below must
    # not be read as "the server told us zero policies". The state pair rides the same rule.
    policies: list[str] | None = None
    state: str | None = None
    edit_session_id: str | None = None
    try:
        client = build_default_client(settings)
        answer = await client.fetch_eval_set(cfg.eval_set_id)
        policies = eval_set_policy_names(answer, cfg.eval_set_id)
        state, edit_session_id = answer.state, answer.edit_session_id
        _record_eval_set_state(cfg.eval_set_id, answer)
    except CreditRefused as exc:
        why = _failure_text(exc)
        print(f'[eval-set] WARNING: could not verify eval set {cfg.eval_set_id} ({why}); proceeding.', file=sys.stderr)
        _record_eval_set_unverified(cfg.eval_set_id, why)
        return None
    except httpx.HTTPStatusError as exc:
        _reraise_outage(exc)  # the read waited out the transient budget; the caller stops the run as unavailable
        status = exc.response.status_code
        if status not in _EVAL_SET_FATAL_STATUSES:
            why = f'HTTP {status}'
            print(f'[eval-set] WARNING: could not verify eval set {cfg.eval_set_id} ({why}); proceeding.', file=sys.stderr)
            _record_eval_set_unverified(cfg.eval_set_id, why)
            return None
        detail = _server_detail(exc.response)
        return (
            f'eval set {cfg.eval_set_id} was rejected by the backend (HTTP {status}) — every scenario would '
            f'fail identically on POST /threads, so nothing is run.'
            + (f'\n       server said: {detail}' if detail else '')
            + '\n       Fix the eval set (regenerate it, or repair its policies_config) and retry, or drop --eval-set-id to run unscored.'
        )
    except Exception as exc:  # noqa: BLE001 — a blip must never refuse a run the box could serve
        _reraise_outage(exc)  # no answer across the transient budget; the caller stops the run as unavailable
        print(f'[eval-set] WARNING: could not verify eval set {cfg.eval_set_id} ({type(exc).__name__}: {exc}); proceeding.', file=sys.stderr)
        _record_eval_set_unverified(cfg.eval_set_id, f'{type(exc).__name__}: {exc}')
    finally:
        # The close is bookkeeping and must never mask the verdict above — nor invent a
        # failure of its own. This helper's whole job is to RETURN an answer about the
        # eval set; a raise here would abort a run over a socket teardown.
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 — best-effort teardown of a probe client
                logger.debug('closing the eval-set pre-flight client failed', exc_info=True)
    if policies is not None and not policies:
        # WHICH remedy, decided from the state rather than assumed: a mid-edit draft can
        # transiently resolve to zero policies, and "regenerate the eval set" is the wrong
        # instruction for a set somebody has open in a session right now.
        remedy = (
            f'\n       An edit session ({edit_session_id or "id unreported"}) is OPEN on this set, so the zero comes '
            'from its mid-edit draft — commit or close the session rather than regenerating the set.'
            if state == _EVAL_SET_DRAFT
            else '\n       Regenerate the eval set (or repair its policies_config) and retry, or drop --eval-set-id to run unscored on purpose.'
        )
        return (
            f'eval set {cfg.eval_set_id} resolves to ZERO policies — nothing would be graded, yet every scenario '
            f'would finalize with pass_rate 0.000 and the run would report N/N succeeded, so nothing is run.' + remedy
        )
    return None


@dataclass(frozen=True, slots=True)
class _PriorRun:
    """What a PREVIOUS invocation's ``run_summary.json`` still vouches for, per row.

    A record rather than a tuple because it only ever grows: every new per-row fact the summary
    carries (grading, early exits, measurement integrity, harness-degraded turns, ceiling
    re-runs) has to survive a partial re-run, and positional unpacking is how the fifth one gets
    silently bound to the fourth one's name.
    """

    scenarios: dict[str, str] = dataclasses_field(default_factory=dict)
    graded: set[str] = dataclasses_field(default_factory=set)
    ended_early: dict[str, str] = dataclasses_field(default_factory=dict)
    ended_early_causes: dict[str, str] = dataclasses_field(default_factory=dict)
    #: The prior rows whose cause is unknown because NO WRITER RECORDED ONE — a summary with no
    #: ``ended_early_causes`` key, or rows an earlier merge already listed. Kept
    #: apart from "the backend stated none": the two absences have different owners.
    ended_early_causes_unrecorded: set[str] = dataclasses_field(default_factory=set)
    measurement_integrity: dict[str, Any] = dataclasses_field(default_factory=dict)
    user_sim_integrity: dict[str, Any] = dataclasses_field(default_factory=dict)
    degraded: dict[str, int] = dataclasses_field(default_factory=dict)
    world_unread: dict[str, Any] = dataclasses_field(default_factory=dict)
    tool_channel_unread: dict[str, list[str]] = dataclasses_field(default_factory=dict)
    mock_unanswered: dict[str, dict[str, int]] = dataclasses_field(default_factory=dict)
    mock_unanswered_unclaimed: dict[str, bool] = dataclasses_field(default_factory=dict)
    ipc_trimmed: dict[str, int] = dataclasses_field(default_factory=dict)
    echoes_dropped: dict[str, dict[str, int]] = dataclasses_field(default_factory=dict)
    turn_cap_reruns: dict[str, Any] = dataclasses_field(default_factory=dict)
    timings: dict[str, Any] = dataclasses_field(default_factory=dict)
    stability: dict[str, Any] = dataclasses_field(default_factory=dict)
    scenario_verdicts: dict[str, str] = dataclasses_field(default_factory=dict)
    scenario_high_severity: dict[str, list[str]] = dataclasses_field(default_factory=dict)
    plan_exhausted_disputed_turns: dict[str, int] = dataclasses_field(default_factory=dict)
    agent_close_preempted_targets: dict[str, str] = dataclasses_field(default_factory=dict)
    #: The prior summary's run-level ``host_effects`` block: the processes that reported it are gone, so a partial re-run
    #: unions it (:func:`_union_host_effects`) rather than dropping every scenario it carries.
    host_effects: dict[str, Any] = dataclasses_field(default_factory=dict)


def _prior_rows(prior: dict[str, Any], key: str, readable: Callable[[Any], bool]) -> dict[str, Any]:
    """The rows of a prior summary's ``key`` map whose SHAPE this version can read.

    VALIDATE, never coerce — the rule every per-row block below is read under, written once so
    a new block cannot be added under a weaker one. ``str(v)`` would accept any shape a future (or
    older) writer might use: a dict row would become the string ``"{'status': 'ok'}"``, which is
    not ``'ok'`` and would count as a FAILURE; a truthy string in a count field would
    carry a degradation claim forward with a nonsense magnitude. A row this version cannot read
    is dropped rather than reinterpreted, and a missing or non-mapping block is no rows at all.

    ``readable`` is per block because the blocks disagree about what a claim IS, and each
    disagreement is load-bearing (see the call sites).
    """
    raw = prior.get(key)
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and readable(v)}


def _merge_prior_scenarios(out: Path, scenarios: dict[str, str]) -> _PriorRun:
    """``scenarios`` overlaid on the scenario map a PREVIOUS invocation left in ``out``, plus
    the ids that invocation recorded as GRADED, the ones it recorded as ENDED EARLY, and the
    per-row measurement-integrity blocks it recorded (all empty when nothing was merged or it
    named none).

    The grading set is returned rather than left behind because merging makes this file
    cumulative, and ``scored`` then describes rows THIS invocation never ran. Both directions
    lie: a scored run followed by an unscored retry labels graded rows "not measured", and an
    unscored run followed by a scored retry labels never-graded rows "measured" — the exact
    mislabelling this contract exists to prevent. The ended-early map rides along for the
    same reason: a carried-forward row whose reason was dropped reads as a conversation that
    ran its course. The integrity map rides along on the same argument — a carried-forward row
    whose breach was dropped reads as a row whose measurement was sound.

    PER-ROW, from ``graded_scenarios``, not from the prior file's ``scored`` boolean. A
    whole-population flag cannot be re-scoped onto the subset of rows that survives a merge,
    and — decisively — a ``scored: true`` states a CLI argument,
    not a measurement. Honoring it would launder that claim into this file's per-row set. A
    prior file that names no graded rows therefore vouches for none.

    A targeted re-run (``--scenario-id``) rewrites this file with only the ids it ran, so
    without merging a 5-of-7 retry into the same ``--out`` would publish ``total: 5`` while
    seven scenario directories sit beside it — and ``build_run_json`` (which counts
    DIRECTORIES) would read seven, so the run's own two artifacts would disagree. Merging
    keeps the summary describing the same population the aggregator will see.

    This invocation's outcome WINS for any id it ran — a scenario that failed before and
    passed now is a pass. Ids only the prior file knew are carried through unchanged.

    Best-effort, but NOT silent. Degrading to this run's map alone reproduces exactly the
    shrinking total this function exists to prevent — a partial re-run publishing ``total: 5``
    beside seven scenario directories — so the degrade says so. Absence of a prior file is the
    one ordinary case (a first run, or a partial run into a fresh ``--out``) and is not worth a
    line; a file that exists and cannot be used is not ordinary."""
    path = out / RUN_SUMMARY_FILENAME
    if not path.exists():
        return _PriorRun(scenarios=dict(scenarios))
    try:
        prior = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning(
            'could not read the previous %s in %s (%s); this run_summary describes only the %d scenario(s) THIS '
            'invocation ran, so its total may be short of the scenario directories present.',
            RUN_SUMMARY_FILENAME,
            out,
            type(exc).__name__,
            len(scenarios),
        )
        return _PriorRun(scenarios=dict(scenarios))
    prior_scenarios = prior.get('scenarios') if isinstance(prior, dict) else None
    if not isinstance(prior_scenarios, dict):
        logger.warning(
            'the previous %s in %s carries no usable "scenarios" mapping; this run_summary describes only the %d '
            'scenario(s) THIS invocation ran, so its total may be short of the scenario directories present.',
            RUN_SUMMARY_FILENAME,
            out,
            len(scenarios),
        )
        return _PriorRun(scenarios=dict(scenarios))
    # VALIDATE the rows, never coerce them. ``str(v)`` accepts any shape a future (or older)
    # writer might use — a dict row becomes the string "{'status': 'ok'}", which is not 'ok' and
    # is therefore counted as a FAILURE. Silently turning someone's passes into failures is a
    # worse outcome than the short total this merge exists to prevent, so an unreadable row is
    # dropped and SAID.
    merged = {k: v for k, v in prior_scenarios.items() if isinstance(k, str) and isinstance(v, str)}
    unusable = len(prior_scenarios) - len(merged)
    if unusable:
        logger.warning(
            'the previous %s in %s carries %d scenario row(s) whose shape this version does not understand; they are '
            'NOT merged, so this run_summary may be short of the scenario directories present.',
            RUN_SUMMARY_FILENAME,
            out,
            unusable,
        )
    prior_graded_raw = prior.get('graded_scenarios')
    prior_graded = {g for g in prior_graded_raw if isinstance(g, str)} if isinstance(prior_graded_raw, list) else set()
    prior_early = _prior_rows(prior, 'ended_early_scenarios', lambda v: isinstance(v, str))
    prior_causes = _prior_rows(prior, 'ended_early_causes', lambda v: isinstance(v, str))
    # A prior with cut rows and no ``ended_early_causes`` key at all never recorded their
    # causes. Those rows' causes are not "none stated" — nothing ever asked — and a merge that
    # carried them into an empty map would let the screen say the backend stated none. Named
    # here (the log) and on the verdict (``ended_early_causes_unrecorded``), and carried through a
    # later merge by reading the same list back.
    unrecorded_raw = prior.get('ended_early_causes_unrecorded')
    prior_unrecorded = {sid for sid in unrecorded_raw if isinstance(sid, str)} if isinstance(unrecorded_raw, list) else set()
    if prior_early and 'ended_early_causes' not in prior:
        prior_unrecorded |= set(prior_early)
        logger.warning(
            'the previous %s in %s predates ended_early_causes: the give-up cause of %d carried row(s) (%s) was never recorded, '
            'so "no cause on record" for them means no writer asked, not that the backend stated none.',
            RUN_SUMMARY_FILENAME,
            out,
            len(prior_early),
            ', '.join(sorted(prior_early)),
        )
    # One step stricter than the rows above: a block without a boolean ``clean`` is not a claim,
    # and carrying it forward would let a shape this version cannot read decide whether a merged
    # run reports its measurement as sound.
    prior_integrity = _prior_rows(prior, 'measurement_integrity', lambda v: isinstance(v, dict) and isinstance(v.get('clean'), bool))
    # ``None`` admitted alongside a bool, unlike its sibling above: this writer records an
    # UNREADABLE user-sim block as ``clean: None`` (see _user_sim_integrity) rather than dropping
    # a stated breach it cannot parse, and requiring a bool here would drop that row one
    # merge later.
    prior_user_sim = _prior_rows(prior, 'user_sim_integrity', lambda v: isinstance(v, dict) and isinstance(v.get('clean'), (bool, type(None))))
    # A count is a COUNT (``bool`` is an ``int`` and is not one).
    prior_degraded = _prior_rows(prior, 'degraded_scenarios', lambda v: isinstance(v, int) and not isinstance(v, bool))
    # Its sibling on the same terms: a row is a claim only with a COUNT of blind reads and the tools
    # that made them, so a partial re-run cannot flip the merged verdict back to clean by dropping a
    # still-broken row, nor carry forward a shape nothing wrote.
    prior_unread = _prior_rows(
        prior,
        'world_unread_scenarios',
        lambda v: isinstance(v, dict) and isinstance(v.get('calls'), int) and not isinstance(v.get('calls'), bool) and isinstance(v.get('tools'), list),
    )
    # Per-RULE counts, so the same count rule applies one level in, and a row left with no
    # readable rule is dropped rather than kept as an empty claim.
    prior_echoes = {
        k: readable_rules
        for k, v in _prior_rows(prior, 'echoes_dropped_scenarios', lambda v: isinstance(v, dict)).items()
        if (readable_rules := {rule: n for rule, n in v.items() if isinstance(rule, str) and isinstance(n, int) and not isinstance(n, bool)})
    }
    prior_reruns = _prior_rows(prior, 'turn_cap_reruns', lambda v: isinstance(v, dict))
    prior_timings = _prior_rows(prior, 'scenario_timings', lambda v: isinstance(v, dict))
    # One step stricter than a bare mapping, on the same reasoning as the integrity blocks
    # above: a row without a readable ``stability`` is not a claim about stability, and
    # carrying it forward would let an unreadable shape decide whether a redraw is licensed.
    prior_stability = _prior_rows(prior, 'stability', lambda v: isinstance(v, dict) and isinstance(v.get('stability'), str))
    merged.update(scenarios)
    return _PriorRun(
        scenarios=merged,
        graded=prior_graded,
        ended_early=prior_early,
        ended_early_causes=prior_causes,
        ended_early_causes_unrecorded=prior_unrecorded,
        measurement_integrity=prior_integrity,
        user_sim_integrity=prior_user_sim,
        degraded=prior_degraded,
        world_unread=prior_unread,
        tool_channel_unread=_prior_rows(prior, 'tool_channel_unread_scenarios', lambda v: isinstance(v, list)),
        # A row is a claim only with a COUNT of unanswered calls, on the rule its neighbours follow.
        mock_unanswered=_prior_rows(
            prior, 'mock_unanswered_scenarios', lambda v: isinstance(v, dict) and isinstance(v.get('calls'), int) and not isinstance(v.get('calls'), bool)
        ),
        mock_unanswered_unclaimed=_prior_rows(prior, 'mock_unanswered_unclaimed_scenarios', lambda v: v is True),
        ipc_trimmed=_prior_rows(prior, 'ipc_trimmed_scenarios', lambda v: isinstance(v, int) and not isinstance(v, bool)),
        echoes_dropped=prior_echoes,
        turn_cap_reruns=prior_reruns,
        timings=prior_timings,
        stability=prior_stability,
        scenario_verdicts=_prior_rows(prior, 'scenario_verdicts', lambda v: isinstance(v, str)),
        scenario_high_severity=_prior_rows(prior, 'scenario_high_severity', lambda v: isinstance(v, list)),
        plan_exhausted_disputed_turns=_prior_rows(prior, 'plan_exhausted_disputed_turns', lambda v: isinstance(v, int) and not isinstance(v, bool)),
        agent_close_preempted_targets=_prior_rows(prior, 'agent_close_preempted_targets', lambda v: isinstance(v, str)),
        # A --trials run's $OUT summary is the composed top, whose `host_effects` already unions trials 2..N (each of which
        # carries its own prior): trial 1 takes back only its own block, or every re-run would sum those trials again.
        host_effects=next((b for b in (prior.get(TRIAL_1_HOST_EFFECTS), prior.get('host_effects')) if isinstance(b, dict)), {}),
    )


def _scenario_was_graded(result: ThreadResult | None) -> bool:
    """Did this scenario produce a MEASUREMENT — at least one policy that actually scored?

    Evidence off the finalize payload, never a re-reading of the CLI flag. Four ways a
    scenario reaches the summary having measured nothing, none of them visible in
    succeeded/failed and all of them reachable:

    * it never finished (no ``ThreadResult`` — ``error.json`` was written instead);
    * the cell map came back empty. The EMPTY MAP is the evidence, never the ``status`` beside
      it: ``'failed'`` means a decisive policy fired, and such a run carries a full set of live
      cells. Gating on it would drop the worst-performing agents first. An untrustworthy world is
      not this case: the backend scores it ``'completed'`` with the world unobserved (world checks
      abstain, transcript checks grade) — see :func:`_untrustworthy_world` for the typed fact;
    * the eval set resolved to no policies, so there are no cells at all (a finalize with
      no ``PolicyManager`` omits the key entirely);
    * every policy ERRORED — the judge broke, not the agent. An errored cell scores no
      signal and never dents ``pass_rate``, so a suite of dead policies reads as a healthy
      1.000 pass (``_write_success_artifacts`` names this hazard).

    A cell that ran counts even when it says ``violated`` or ``inconclusive``: those are
    verdicts about the agent. Only "the eval never landed" is ungraded. Skipped policies
    (``applicability_guard`` / ``flow_filter``) did not run on this scenario, so they carry
    no measurement either — a scenario whose every policy was skipped was not graded.
    """
    return bool(_graded_policies(result))


def _untrustworthy_world(result: ThreadResult | None) -> str | None:
    """Why this run's WORLD cannot be scored, or ``None`` — the typed fact ``status`` stood in for.

    ``finalize.status`` does not carry it: ``'failed'`` means only a scored run carrying a DECISIVE
    violation.
    A run over an untrustworthy world — a crash marker, or a proxy failure the world corroborates —
    is scored with the world unobserved: ``'completed'`` with ``world_state_trustworthy: false``,
    its world checks abstaining and its transcript checks grading. Reading ``'failed'`` as
    "unscored" is a catastrophic red erasing its own measurement: the run would report
    ``scored: false`` and render "this run was NEVER SCORED — the verdicts below are
    placeholders" over a full set of landed cells. That would recur on every run with a
    catastrophic red, i.e. the run a customer most needs to read.

    So the untrustworthy world gets its own field, from the two places that actually say it: the
    backend's ``world_state_trustworthy`` block, and the child's ``_finalize_error`` marker (the
    snapshot never materialized). Both name a REASON, which ``status`` never could.
    """
    if result is None:
        return None
    per = result.finalize.policy_evaluation_results or {}
    if isinstance(per, dict) and per.get('world_state_trustworthy') is False:
        return str(per.get('world_state_distrust_reason') or 'the backend did not certify the final world of this run')
    return _finalize_error_message(result.finalize.final_world_state or {}, result.tool_events or {})


def _policy_cells(result: ThreadResult | None) -> dict[str, dict[str, Any]]:
    """The per-policy CELLS a finalize landed — the dict-shaped ones only — else ``{}``.

    The one reading of "which cells landed", shared by the graded and the errored views below so
    they partition the same set. A cell that is not a mapping (``None``, ``True``) is not a
    verdict this version can read and is neither graded nor errored: counting it as graded would
    make a malformed payload read as a measurement.

    A PURE read of the payload, gated on NEITHER ``finalize.status`` NOR the typed
    untrustworthy-world fact. Both gates erase the same thing, and the second one is not a
    smaller version of the first: the backend scores every distrusted world — a crash marker or a
    corroborated proxy failure with the world unobserved (world checks abstain, transcript checks
    grade), the backend's ZERO-ENTITY verdict over the world as read — and each
    rides on a ``status: completed`` payload with its cells populated and a real ``pass_rate``
    beside it. Dropping those cells makes the reds vanish from the rollup, ``scored`` go
    ``false`` under a backend that scored the run, and :func:`_policy_evidence` — reading the
    same empty map — stamp every judged red ``deterministic``.

    The distrust is not swallowed, it is REPORTED as its own typed fact, which is what
    :func:`_untrustworthy_world` exists for: ``untrustworthy_world`` on the row's transcript,
    ``untrustworthy_scenarios`` in the run summary, and a line on the operator's screen from
    :func:`_record_untrustworthy`. A caveat beside the measurement, never an erasure of it.
    """
    if result is None:
        return {}
    cells = (result.finalize.policy_evaluation_results or {}).get('evaluation_results')
    if not isinstance(cells, dict):
        return {}
    return {str(name): cell for name, cell in cells.items() if isinstance(cell, dict)}


def _graded_policies(result: ThreadResult | None) -> frozenset[str]:
    """WHICH policies produced a measurement on this attempt — the names behind
    :func:`_scenario_was_graded`, under its rules, so the two can never disagree about what
    counts as graded. The mass-failure shed reads the names (a row red on every one of them,
    identically across rows, is a reading about the apparatus); the boolean reads their count;
    the per-policy stability record reads them to know which reds the retry could measure."""
    cells = _policy_cells(result)
    if not cells:
        return frozenset()
    per = result.finalize.policy_evaluation_results or {} if result is not None else {}
    # Both name lists are ``None`` when the server never sent them (the backend may send no
    # ``errored_policies`` list — the per-cell ``status`` is where an errored
    # judge shows up). Absence is not a measurement, so it contributes no names rather than
    # an empty claim; the per-cell check below is what actually catches a dead suite.
    errored = _errored_policies(result)
    skipped = set(_policy_name_list(per, 'skipped_policies') or [])
    return frozenset(name for name in cells if name not in errored and name not in skipped)


def _errored_policies(result: ThreadResult | None) -> frozenset[str]:
    """WHICH cells' JUDGE broke on this attempt — ``status: 'error'`` on the cell, or named in
    ``errored_policies``. The complement :func:`_graded_policies` subtracts, read from the same
    cells so a policy is never both. The mass-failure shed reads it for the judge-fleet death: a
    finalize that completed with every cell errored is a dead apparatus wearing a clean status."""
    cells = _policy_cells(result)
    if not cells:
        return frozenset()
    per = result.finalize.policy_evaluation_results or {} if result is not None else {}
    named = set(_policy_name_list(per, 'errored_policies') or [])
    return frozenset(name for name, cell in cells.items() if name in named or cell.get('status') == 'error')


def _measurement_integrity(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, dict[str, Any]]:
    """Per row, the backend's own ``measurement_integrity`` block — READ, never re-derived.

    The backend computes it at ``/finalize`` by joining the channels only it can see (an errored
    judge, a raising applicability_guard, a crashed per-turn pass, a broken deterministic
    target check) and rides it on ``policy_evaluation_results``, which this runner already
    persists verbatim into ``transcript.json``. Re-deriving any of that here would be a second
    implementation of a fact the run already stated, and the two would drift.

    Why it must reach ``run_summary.json`` at all: none of those failures dents ``pass_rate``
    or ``aggregate_score`` — an errored cell scores nothing — so a scenario whose whole target
    check died reads as a clean pass everywhere else (``aggregate_score: 1.0`` with ``failed: 0``).

    Rows carrying NO block are omitted rather than defaulted to clean: an older backend sends
    none, and inventing ``clean: true`` for it is exactly the false green this exists to close.
    """
    out: dict[str, dict[str, Any]] = {}
    for scenario_id, result in pairs:
        if result is None:
            continue
        block = (result.finalize.policy_evaluation_results or {}).get('measurement_integrity')
        # ``clean`` present and boolean is what makes this a CLAIM the backend checked. A block
        # without it is a shape this version does not understand, and reading it either way
        # would be a guess.
        if isinstance(block, dict) and isinstance(block.get('clean'), bool):
            out[scenario_id] = block
    return out


def _eval_revisions(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, str]:
    """Per row, the eval set revision finalize stamped on it — READ, never derived.

    The backend rides ``eval_revision`` on ``policy_evaluation_results`` (the content digest over the
    set's per-policy fingerprints). It is what lets a saved run say WHICH judges graded
    it: an in-place rubric edit keeps ``eval_set_id`` and moves this. Rows an older backend
    scored carry none and are omitted, never defaulted.

    The revision is the AUTHORITATIVE attempt's — attempt 1 everywhere except the superseded
    invalid run, where the retry replaces it (``authoritative_attempt`` on the stability row). The
    per-ATTEMPT stamps are :func:`_marks_comparability`'s."""
    out: dict[str, str] = {}
    for scenario_id, result in pairs:
        revision = _eval_revision_of(result)
        if revision is not None:
            out[scenario_id] = revision
    return out


def _eval_revision_of(result: ThreadResult | None) -> str | None:
    """The eval set revision finalize stamped on ONE attempt, or ``None`` when it stamped none.

    Split out of :func:`_eval_revisions`: the stability row needs the same read PER ATTEMPT (see
    :func:`_marks_comparability`).
    """
    if result is None:
        return None
    revision = (result.finalize.policy_evaluation_results or {}).get('eval_revision')
    return revision if isinstance(revision, str) and revision else None


def _user_sim_integrity(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, dict[str, Any]]:
    """Per row, the backend's ``user_sim_integrity`` block — READ, never re-derived.

    The OTHER half of "was this run's apparatus sound". The backend folds the user simulator's own
    per-turn fidelity anomalies into
    this block at ``/finalize`` and rides it on ``policy_evaluation_results`` — the same payload
    this runner already persists verbatim.

    Why it belongs in the summary: an unclean user simulator means the TRANSCRIPT is unfaithful
    to the scenario that was authored, so every verdict scored on it is a verdict about a
    conversation nobody designed. A summary that published ``measurement_clean: true`` while a
    transcript's ``user_sim_integrity.clean`` was ``false`` would misstate the run.

    Kept as its OWN per-row map rather than folded into ``measurement_integrity``: the backend keeps
    them apart deliberately because they have different owners and different remedies — a broken
    judge is an eval fix, a simulator that departed from its schedule is a scenario/simulator
    fix and no eval change repairs it. The run-level roll-up is where they meet (see
    :func:`_measurement_clean`), because "was anything here left unmeasured" does not care which
    side the defect was on.

    Rows carrying no block are OMITTED rather than defaulted to clean, as in
    :func:`_measurement_integrity`.

    A block this version cannot READ is different, and is the one thing not shared with its
    sibling. Dropping it would make the roll-up's defensive ``clean is not True`` unreachable —
    every surviving block would hold a real bool — so a backend shipping ``"clean": "false"`` would
    have a STATED BREACH silently discarded, while the same row's ``measurement_integrity`` omission at
    least shows up as a lower ``reported`` count. Such a row is therefore kept as UNKNOWN —
    ``clean: None`` with ``state: unchecked``, the producer's own "not measured" value — and the
    payload preserved verbatim under ``unreadable_block`` for whoever has to reconcile it. Unknown
    is neither good news nor a breach: the run-level fold reads it as an unmeasured channel
    (:func:`_measurement_unchecked_because`), never as ``measurement_clean: false``.
    """
    out: dict[str, dict[str, Any]] = {}
    for scenario_id, result in pairs:
        if result is None:
            continue
        block = (result.finalize.policy_evaluation_results or {}).get('user_sim_integrity')
        if block is None:
            continue
        # ``clean`` is a bool, OR the producer's own third value: ``None`` beside
        # ``state: unchecked`` is "the channel could not be read" (the backend finalize), not a
        # shape this version cannot parse — filing it as unreadable would strip the ``state``
        # and the per-turn map that say WHY.
        if isinstance(block, dict) and (isinstance(block.get('clean'), bool) or (block.get('clean') is None and block.get('state') == 'unchecked')):
            out[scenario_id] = block
            continue
        logger.warning(
            'scenario %s: user_sim_integrity has a shape this version cannot read (%s); recording it as UNKNOWN '
            '(unchecked) rather than dropping it or reading it as either clean or breached.',
            scenario_id,
            type(block).__name__,
        )
        out[scenario_id] = {'clean': None, 'state': 'unchecked', 'unreadable_block': block}
    return out


#: The bucket a scenario declaring no family lands in. NAMED, never dropped: a gradient whose
#: unclassified scenarios are silently absent is a gradient with a hole in its denominator.
UNCLASSIFIED_FAMILY = '(unclassified)'


def _family_row(result: ThreadResult | None) -> dict[str, Any] | None:
    """One scenario's contribution to the family gradient, or ``None`` when it has none.

    Split from the fold below because the two run strategies hold their results in different
    shapes — one has every ``ThreadResult`` at the end, the other only sees each as its task
    completes — and a second copy of the projection is how they drift into disagreeing about
    what "passed" means.

    ``None`` for a row that never produced a result, and for one whose backend sent no
    ``scenario_dimensions``: an older backend saying nothing must not be bucketed as
    "declared no family", because those are different facts.
    """
    if result is None or not _scenario_was_graded(result):
        # An UNGRADED row measured nothing, so it belongs to no family at all — not even to a
        # family's population. (The ``verdicted`` gate below keeps an unmeasured row out of the
        # RATE; this is the stronger case, where there is no result to read a family from.)
        return None
    per = result.finalize.policy_evaluation_results or {}
    dimensions = per.get('scenario_dimensions')
    if not isinstance(dimensions, dict):
        return None
    flows = dimensions.get('target_flows')
    rate = per.get('pass_rate')
    # Not derived from ``violated_policies`` (empty list ⇒ passed): that any-violated derivation
    # gives a false green, because a scenario whose judges all errored has no violated policies.
    # The backend owns the verdict; this reads it, so this gradient, the observability rollup and the sink
    # summary cannot disagree about what "passed" means. (The terminal headline tallies the same
    # reader, per row, through :func:`_scenario_verdict_ids`.)
    verdict = read_scenario_verdict(per)
    return {
        # A scenario in two families counts in BOTH — flows overlap by design, and forcing a
        # single bucket would need a tie-break rule this has no basis to invent.
        'families': [str(f) for f in flows] if isinstance(flows, list) and flows else [UNCLASSIFIED_FAMILY],
        'passed': verdict.status == 'passed',
        # Did the row reach a goal verdict AT ALL? A ``not_evaluated`` / ``unclean`` /
        # ``unknown`` row is neither a pass nor a fail, so it leaves the family's rate the way
        # ``incomplete`` does — counted, named, and out of the fraction. Without this a run
        # scored by a backend that publishes no verdict block would render every family at 0%,
        # a manufactured red.
        'verdicted': verdict.conclusive,
        # A row the TIME BUDGET cut is not a pass. Its evals are low for obligations no turn
        # reached, so folding it into ``passed`` reports a family of timed-out scenarios at
        # 100% — counted apart, the way the run verdict counts it apart from both neighbours.
        'incomplete': _row_outcome(result) == 'incomplete',
        'pass_rate': float(rate) if isinstance(rate, (int, float)) and not isinstance(rate, bool) else None,
    }


def _scenario_verdict_ids(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, str]:
    """Per row that returned a result, its house verdict status (:func:`read_scenario_verdict`)."""
    return {sid: read_scenario_verdict(result.finalize.policy_evaluation_results).status for sid, result in pairs if result is not None}


def _scenario_high_severity_ids(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, list[str]]:
    """Per row whose verdict served one, the backend's ``high_severity_violations`` (:func:`read_scenario_verdict`)."""
    return {
        sid: list(high)
        for sid, result in pairs
        if result is not None and (high := read_scenario_verdict(result.finalize.policy_evaluation_results).high_severity_violations) is not None
    }


def _family_gradient(rows: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    """Per-family pass rates for the run report.

    A suite verdict of ``0/7`` says only "everything failed". A GRADIENT localizes it: 0% in
    one family and 100% everywhere else is a defect with an address, and a flat rate across
    families is a different — usually worse — story. The families are the scenario's own
    declared ``target_flows``, forwarded by the backend on
    ``policy_evaluation_results.scenario_dimensions``; nothing here interprets them, because
    which flows exist is per-customer data and the generic rule is only "group by what the
    scenario declared".

    ``coverage`` is stated beside the buckets because a gradient over 2 of 7 rows and a
    gradient over 7 of 7 look identical once the rates are computed.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    reported = 0
    population = 0
    for row in rows:
        population += 1
        if row is None:
            continue
        reported += 1
        for name in row['families']:
            buckets.setdefault(name, []).append(row)
    families: dict[str, Any] = {}
    for name, bucket in sorted(buckets.items()):
        # The DENOMINATOR is the rows that reached a goal verdict over a whole conversation —
        # not every row in the bucket: counting ``incomplete`` or ``not_evaluated`` rows there
        # would read a family of timed-out scenarios as 0%. A rate divides by what was decided;
        # what was not decided is reported beside it, which is what the two counts below do.
        decided = [row for row in bucket if row['verdicted'] and not row['incomplete']]
        passed = sum(1 for row in decided if row['passed'])
        # Only the rows that reported one: a family whose backend sent no float must read
        # "no rate", not a mean pulled down by the rows that had nothing to contribute.
        rates = [row['pass_rate'] for row in bucket if row['pass_rate'] is not None]
        families[name] = {
            'scenarios': len(bucket),
            'passed': passed,
            'incomplete': sum(1 for row in bucket if row['incomplete']),
            # Named, never folded into the rate: "this family reached no verdict" and "this
            # family failed" are opposite findings.
            'not_evaluated': sum(1 for row in bucket if not row['verdicted'] and not row['incomplete']),
            # ``None``, not 0.0, when nothing in the bucket was decided — a family whose rows
            # were all unmeasured has no rate, and a 0.0 there reads as a total failure.
            'scenario_pass_rate': round(passed / len(decided), 3) if decided else None,
            'mean_policy_pass_rate': round(sum(rates) / len(rates), 3) if rates else None,
        }
    # A numerator with no denominator misleads: a gradient over 2 of 7 rows and one over 2 of 2 look
    # identical once the rates are computed, which is the same reason ``measurement_coverage``
    # and ``user_sim_coverage`` both state their population.
    return {'families': families, 'coverage': {'reported': reported, 'population': population}}


def _turn_cap_rerun_allowed(cfg: _RunConfig) -> bool:
    """May this run spend a ceiling-hit re-run? Asked in TWO frames, so it is answered once.

    The scenario driver spends the budget; the vendor executor sizes its task timeout off the
    largest ceiling any attempt can reach. If those two disagreed about whether a re-run is
    possible, the vendor timeout would be sized for an attempt count the runner does not use —
    and the vendor's timeout requeues instead of cancelling, which is the race the headroom
    exists to win.
    """
    return cfg.rerun_on_turn_cap and cfg.run_phase != 'qualification'


def _resolve_scenario_budget(cfg: _RunConfig, settings: Settings) -> float:
    """The per-scenario time budget in force, from the ONE place that knows both halves.

    ``_settings_from_args`` always fills the field (flag > env > phase default), so the fallback
    is only reached by a library caller that built ``Settings`` directly — and it is the phase
    default rather than a bare constant, because the phase is exactly what this frame knows and
    ``NetworkSettings`` does not.
    """
    return settings.network.scenario_timeout or default_scenario_time_budget(cfg.run_phase)


def _worst_case_scenario_ceiling(cfg: _RunConfig, settings: Settings) -> float:
    """The largest hard ceiling ANY attempt of one scenario can run under.

    A scenario is not one attempt. A turn-cap re-run doubles both the turn allowance and the
    time budget, and re-derives its ceiling from the doubled budget — so the vendor executor's
    per-task timeout has to clear THAT, not the first attempt's. The vendor's timeout does not
    cancel: it requeues at the same priority with no attempt counter, so if it wins the race an
    orphaned run keeps going while a second attempt deletes the first's artifacts.
    """
    budget = _resolve_scenario_budget(cfg, settings)
    if _turn_cap_rerun_allowed(cfg):
        budget *= TURN_CAP_RERUN_MULTIPLIER
    ceiling = derive_scenario_hard_ceiling(budget, settings.network)
    if cfg.stability_retry_enabled and _RESAMPLES_THE_AGENT[cfg.stability_sample_kind]:
        # An agent re-run is a SECOND full attempt inside the SAME vendor task, so the task's
        # timeout has to clear both. It is not a corner case: an ``incomplete`` row spent its
        # whole budget by definition and then runs the retry on top. Left un-sized, the vendor
        # timeout fires almost every time — and it does not cancel, it REQUEUES, which re-enters
        # _drive_one_scenario at the top level while the orphan is still writing the same
        # scenario folder. That is the exact race this ceiling exists to win.
        ceiling *= 2
    elif cfg.stability_retry_enabled:
        # A judge replay is no second attempt: one submit, then a poll bounded by the same
        # wallclock budget the finalize poll has. Doubling for it would silently double every
        # full run's vendor timeout.
        ceiling += settings.network.http_timeout + settings.network.finalize_poll_max_wallclock_seconds
    return ceiling


#: The row statuses ``run_summary.json`` reports, and the ONE closed set of them. Closed, unlike
#: the ``ended_early`` vocabulary next door, because nothing on the wire mints these: they are
#: derived in-process from a local ThreadResult, so there is no older half to raise on a value a
#: newer half added. Closing it is what makes the counts a partition rather than a guess — every
#: count site can be exhaustive, and a fourth outcome becomes a type error at each of them
#: instead of silently landing in the success column.
RowOutcome = Literal['ok', 'failed', 'incomplete']


#: Scenarios THIS RUN never measured because the SUITE refused to open a thread for them.
#: A module-level handoff, the same shape ``_LAST_EGRESS_RECORD`` uses and for the
#: same reason: the refusal is classified where the exception is (``_write_error``) and the row
#: status is computed later, in a frame that holds only a scenario id and a ``None`` result.
_QUARANTINED_ROWS: set[str] = set()

#: Scenarios THIS RUN lost to a HARNESS fault (``error.json`` ``kind: harness_*``, the typed class the
#: raiser tags — the SDK re-raised it rather than scoring it). Never scored, so ``not_valid`` in the
#: run's verdict tally, never an agent failure. Same handoff as :data:`_QUARANTINED_ROWS`.
_HARNESS_FAULT_ROWS: set[str] = set()

#: Scenarios THIS RUN stopped before they could finish, because the run was aborted: never started, cancelled mid-flight, or
#: the one whose refusal caused the abort. They measured nothing, so they are ``incomplete`` rows and never failures. Same
#: handoff as :data:`_QUARANTINED_ROWS`.
_NOT_RUN_ROWS: set[str] = set()


def _row_outcome(result: ThreadResult | None, scenario_id: str | None = None) -> RowOutcome:
    """The run-summary row status for one attempt: ``ok`` / ``failed`` / ``incomplete``.

    THE one place the mapping lives, because the two run strategies compute their outcome maps
    in different frames and a second copy is how they drift.

    ``incomplete`` is its own status rather than a flavour of either neighbour. It is not
    ``failed``: the scenario ran, produced artifacts and was graded, and counting it as a
    failure would make a slow flow read as a broken one — and, through the exit code, would
    block the quality loop the budget exists to keep moving. It is not ``ok`` either: the
    dialog was cut, so its evals are low for obligations no turn ever reached, and a reader who
    sees ``ok`` has no way to know the measurement is partial.

    It holds a SECOND shape, and the two share the property the bucket is named for: neither
    passed nor failed. A QUARANTINED scenario (``scenario_id`` in :data:`_QUARANTINED_ROWS`) never
    ran at all — the backend refused to open a thread because the scenario's own reference path
    cannot pass its own eval set. Reporting that as ``failed`` would make the suite DECLINING to
    measure byte-identical, in the file every consumer of a finished run reads, to the agent
    failing: the exact collapse the typed refusal exists to prevent. Unlike the
    time-budget shape it carries no artifacts and no grading, so it joins neither
    ``graded_scenarios`` nor the family gradient — it is a row that took no measurement.

    The EXIT CODE is deliberately untouched: the gather strategy counts failures from the same
    truthiness (``not ok``), and a run that could not measure a scenario should still exit
    nonzero, because something does need fixing. What changes is the CLAIM the row makes about
    the AGENT, not whether the run reports a problem.
    """
    if result is None:
        return 'incomplete' if _unmeasured_by_design(scenario_id, result) else 'failed'
    # A dialog the harness or its infrastructure CUT (the time budget, the per-turn wall, the
    # agent's provider) landed and was graded to the turn it reached: neither passed nor failed.
    return 'incomplete' if result.ended_early in HARNESS_CUT_REASONS else 'ok'


def _not_run(scenario_id: str | None, result: ThreadResult | None) -> bool:
    """Was this row STOPPED by the run's abort before it could finish? Like a quarantined row it took no measurement, so it is
    ``incomplete`` and neither a failure nor a crash."""
    return result is None and scenario_id is not None and scenario_id in _run_ledgers().not_run


def _unmeasured_by_design(scenario_id: str | None, result: ThreadResult | None) -> bool:
    """A row that took no measurement through no fault of the agent or the apparatus: refused by the suite, or stopped by the abort."""
    return _quarantined(scenario_id, result) or _not_run(scenario_id, result)


def _quarantined(scenario_id: str | None, result: ThreadResult | None) -> bool:
    """Was this row REFUSED by the suite before it ran? The one reading, shared by the
    row status and the mass-failure census: a quarantined row has no thread, no apparatus and no
    measurement, so it is ``incomplete`` there and NOT a crash here."""
    return result is None and scenario_id is not None and scenario_id in _run_ledgers().quarantined


def _settle_unlanded_rows(outcomes: dict[str, RowOutcome]) -> None:
    """Re-read every row the experiment strategy still holds as ``failed`` through :func:`_row_outcome`.

    The gather strategy computes each row's outcome from its result AND its scenario id in one
    call; the experiment strategy seeds every row ``failed`` before dispatch and overwrites only
    the rows a completed task reaches, so a scenario the SUITE quarantined mid-run would keep the
    seed value and the two strategies would disagree on the one row whose classification exists to be
    visible. Settled after the tasks, from the same function, so there is still one mapping.
    """
    for scenario_id, outcome in outcomes.items():
        if outcome == 'failed':
            outcomes[scenario_id] = _row_outcome(None, scenario_id)


def _ended_early_ids(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, str]:
    """The rows whose dialog ended for something other than the user reaching their goal, mapped
    to the typed reason.

    NOT gated on anything the way :func:`_graded_ids` is: this is a fact the orchestrator
    reports about a conversation it ran, not a claim about a measurement. Value-agnostic on
    purpose — a reason added to the vocabulary rides this with no change here. A scenario that
    never produced a result contributes nothing — its ``error.json`` is the record.
    """
    return {scenario_id: result.ended_early for scenario_id, result in pairs if result is not None and result.ended_early}


def _plan_exhausted_disputed_turns(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, int]:
    """Per ``plan_exhausted`` row whose LAST simulated-user message pushed back (its own typed
    ``raw_reply.pushed_back``), the 0-based turn of the agent reply that message disputed.

    The stop ends the thread on the first turn after the schedule is spent whatever the user said
    there, so a push-back ending is a disputed agent claim, not a spent schedule."""
    out: dict[str, int] = {}
    for scenario_id, result in pairs:
        if result is None or result.ended_early != PLAN_EXHAUSTED_REASON:
            continue
        info = getattr(result.finalize, 'per_turn_user_step_info', None)
        try:
            last_turn = max(int(key) for key in info) if isinstance(info, dict) and info else None
        except (TypeError, ValueError):
            last_turn = None
        last = info.get(str(last_turn)) if last_turn is not None and isinstance(info, dict) else None
        reply = last.get('raw_reply') if isinstance(last, dict) else None
        if last_turn and isinstance(reply, dict) and reply.get('pushed_back') is True:
            out[scenario_id] = last_turn - 1
    return out


def _agent_close_preempted_targets(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, str]:
    """Per row the AGENT closed before the scenario's target beat arose, that beat — the backend's own
    ``dialog_outcome.preempted_target``. Absent on an older backend, so such a row keeps its route."""
    out: dict[str, str] = {}
    for scenario_id, result in pairs:
        per = result.finalize.policy_evaluation_results if result is not None else None
        outcome = per.get('dialog_outcome') if isinstance(per, dict) else None
        target = outcome.get('preempted_target') if isinstance(outcome, dict) else None
        if isinstance(target, str) and target:
            out[scenario_id] = target
    return out


def _ended_early_causes(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, str]:
    """The rows whose simulated user STATED why it gave up, mapped to that cause — its own map
    beside :func:`_ended_early_ids` because the reason's map is ``{scenario: reason}`` on every
    consumer's disk and a richer value there would break them. Same rule otherwise: relayed,
    value-agnostic, and a row with no stated cause contributes nothing rather than a blank.

    Gated on the REASON, like the artifact writer: a cause beside no ``ended_early`` explains
    nothing, would land in this map and never in the reason map, and would render nowhere — so it
    is dropped and the row named, rather than filed where nothing reads it."""
    causes: dict[str, str] = {}
    for scenario_id, result in pairs:
        if result is None or not result.ended_early_cause:
            continue
        if not result.ended_early:
            logger.warning(
                'scenario %s carries ended_early_cause=%r with no ended_early reason — a cause of nothing; dropped from ended_early_causes.',
                scenario_id,
                result.ended_early_cause,
            )
            continue
        causes[scenario_id] = result.ended_early_cause
    return causes


def _degraded_ids(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, int]:
    """The rows carrying turns the HARNESS emptied, mapped to HOW MANY.

    A client-side measurement-integrity fact, and the one the backend structurally cannot
    report: ``/finalize`` sees a well-formed conversation whose agent turns happen to be empty,
    so its own ``measurement_integrity`` block says ``clean: true`` while some of the graded
    turns carry a reply the agent never gave (``ChildProcess.degraded_turns``).

    Same posture as its sibling ``mock_unanswered`` (tool answers the runner stubbed): the
    runner knows the DATA was partly manufactured even when the apparatus that scored it worked
    perfectly. A count rather than a flag — one emptied turn in twelve and twelve in twelve are
    different runs, and only the second was no measurement of the agent at all (the orchestrator
    marks that one ``ended_early='adapter_degraded'``).
    """
    return {scenario_id: len(result.degraded_turns) for scenario_id, result in pairs if result is not None and result.degraded_turns}


#: The call statuses that mean the tool never answered — a dispatch error, or a harness that
#: gated the tool by name. Neither is a READ, so neither can be an empty one.
_NOT_A_DISPATCHED_CALL: frozenset[str] = frozenset({'error', *TRAIL_NOT_DISPATCHABLE_STATUSES})


def _comparable_values(value: object) -> set[str]:
    """Every scalar leaf under ``value``, in the ONE spelling an argument and a seeded field are compared in.

    Strings case-folded and whitespace-collapsed, numbers by their text; bools, ``None`` and empty
    strings dropped (a flag or an absent value names no record). No length floor and no
    script-specific rule: a reformatted value (``"premium economy"`` for ``premium_economy``) is
    simply not the seeded value, which only ever costs a detection, never manufactures one.
    """
    found: set[str] = set()
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list | tuple):
            pending.extend(item)
        elif isinstance(item, int | float) and not isinstance(item, bool):
            found.add(repr(item))
        elif isinstance(item, str) and (text := ' '.join(item.split()).casefold()):
            found.add(text)
    return found


def _empty_read(event: Mapping[str, Any]) -> bool:
    """Whether a tool call's response is a zero-row shape — the plugin's classifier (``_zero_result``), read
    off the recorded ``response`` (the key both the mock log and the recovery merge write), JSON text
    decoded first. Prose is content, never an empty read."""
    raw = event.get('response')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return False
    return _zero_result(raw)


#: The schema env-gen ships beside the harness, and where it pins each tool's judged record (``tool_strategies``).
_STATE_SCHEMA_FILENAME = 'scenario_state_schema.yaml'

#: ``{tool: {entity_type: {field: values}}}`` — the rows of a type a reader returns, by a kind or lifecycle field.
ReaderScopes = Mapping[str, Mapping[str, Mapping[str, frozenset[str]]]]
#: The ``ReaderScopes`` key (never an entity type) holding ``{arg: {'<type>.<field>', ...}}``: the stored field(s) a
#: ``filter`` / ``reference`` input is compared with (the backend's ``inputs.<arg>.target_field``).
_ARG_TARGETS = '@args'
#: The ``ReaderScopes`` key holding ``{arg: {'<type>.<field>', ...}}``: the targets that are the type's OWN key (its
#: declared ``id_field``, else ``id``). A value found there names that one record whatever the other REFERENCES ask.
_KEY_TARGETS = '@keys'
#: The ``ReaderScopes`` key holding ``{arg: frozenset()}`` for each targeted input whose role is ``reference``.
_REFERENCE_ARGS = '@refs'


def _reader_scopes(workdir: Path) -> dict[str, dict[str, dict[str, frozenset[str]]]]:
    """Each tool's declared ``scope`` off the shipped ``scenario_state_schema.yaml``, in :func:`_comparable_values`' spelling.

    The backend records it per tool (``tool_strategies.tools.<tool>.scope: {'<type>.<field>': [values]}``): a read returns
    only the rows of the type whose field holds one of the values (a reservation slot, a submitted invoice). ``{}``
    for a schema that is absent, unreadable or declares none, and a value that has no comparable spelling (a bool) is
    left out, so an undecidable scope never narrows anything.

    The types a tool's contract says it touches (``tool_contracts.<tool>``: ``reads.entities`` ∪ ``writes.entities``,
    plus its scoped types) land as a scope on EVERY type, under the ``''`` key: a record of any other type is one the
    call never sees. A tool with no contract, or an empty ``reads.entities``, gets none. That scope also carries, under
    the field ``''``, the schema's type vocabulary (declared entity names ∪ every contract's and scope's types): a record
    whose type is spelled outside it (``OrderLine`` for ``order_line``) is undecidable, so it stays comparable.

    A ``filter`` / ``reference`` input that names the stored field(s) it is compared with (``inputs.<arg>.target_field``,
    one ``'<type>.<field>'`` or a list) lands under :data:`_ARG_TARGETS`, only when every path is one the tool ``reads``
    (``<type>.<field>`` or ``<type>.*``); any other target is undecidable and narrows nothing.
    """
    import yaml  # noqa: PLC0415 - read once per run, beside the summary

    try:
        schema = yaml.safe_load((workdir / _STATE_SCHEMA_FILENAME).read_text(encoding='utf-8'))
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    strategies = schema.get('tool_strategies') if isinstance(schema, dict) else None
    tools = strategies.get('tools') if isinstance(strategies, dict) else None
    scopes: dict[str, dict[str, dict[str, frozenset[str]]]] = {}
    for tool, facts in (tools or {}).items() if isinstance(tools, dict) else ():
        scope = facts.get('scope') if isinstance(facts, dict) else None
        for path, values in (scope or {}).items() if isinstance(scope, dict) else ():
            entity_type, _, field_name = str(path).partition('.')
            comparable = frozenset(_comparable_values(values))
            if field_name and comparable:
                scopes.setdefault(str(tool), {}).setdefault(entity_type, {})[field_name] = comparable
    contracts = schema.get('tool_contracts') if isinstance(schema, dict) else None
    declared = schema.get('entities') or schema.get('entity_types') if isinstance(schema, dict) else None
    vocabulary = _comparable_values([list(declared or {}) if isinstance(declared, dict) else [], *(list(scope) for scope in scopes.values())])
    touched_by_tool: dict[str, set[str]] = {}
    for tool, contract in contracts.items() if isinstance(contracts, dict) else ():
        sides = [contract.get(side) if isinstance(contract, dict) else None for side in ('reads', 'writes')]
        reads, writes = (side.get('entities') if isinstance(side, dict) else None for side in sides)
        if isinstance(reads, list) and (read_types := _comparable_values(reads)):
            touched = read_types | _comparable_values(writes if isinstance(writes, list) else []) | _comparable_values(list(scopes.get(str(tool), {})))
            touched_by_tool[str(tool)] = touched
            vocabulary |= touched
    for tool, touched in touched_by_tool.items():
        scopes.setdefault(tool, {})[''] = {'_entity_type': frozenset(touched), '': frozenset(vocabulary)}
    for tool, facts in (tools or {}).items() if isinstance(tools, dict) else ():
        reads = facts.get('reads') if isinstance(facts, dict) else None
        read = {path for path in reads if isinstance(path, str)} if isinstance(reads, list) else set()
        inputs = facts.get('inputs') if isinstance(facts, dict) else None
        for arg, role in inputs.items() if isinstance(inputs, dict) else ():
            target = role.get('target_field') if isinstance(role, dict) and role.get('role') in ('filter', 'reference') else None
            paths = [target] if isinstance(target, str) else target if isinstance(target, list) else []
            if paths and all(isinstance(p, str) and p.partition('.')[2] and {p, f'{p.partition(".")[0]}.*'} & read for p in paths):
                scopes.setdefault(str(tool), {}).setdefault(_ARG_TARGETS, {})[str(arg)] = frozenset(paths)
                if role.get('role') == 'reference':
                    scopes[str(tool)].setdefault(_REFERENCE_ARGS, {})[str(arg)] = frozenset()
                keys = frozenset(p for p in paths if p.partition('.')[2] == _key_field(declared, p.partition('.')[0]))
                if keys:
                    scopes[str(tool)].setdefault(_KEY_TARGETS, {})[str(arg)] = keys
    return scopes


def _key_field(declared: object, entity_type: str) -> str | None:
    """The field a declared type is keyed by: its ``id_field`` (``None`` when declared null), else ``id``."""
    body = declared.get(entity_type) if isinstance(declared, dict) else None
    return body.get('id_field') if isinstance(body, dict) and 'id_field' in body else 'id'


def _in_reader_scope(record: object, scope: Mapping[str, Mapping[str, frozenset[str]]]) -> bool:
    """Whether a reader with ``scope`` could return this seeded record: no scoped field of its type holds a value the
    reader filters out, and its type is one the tool touches (the ``''`` scope). A record without the field, or of an
    unscoped type, stays in; so do an untyped record and one whose type the schema's vocabulary does not spell."""
    if not isinstance(record, dict):
        return True
    entity_type = str(record.get('_entity_type') or '')
    every_type = scope.get('', {})
    known = bool(entity_type) and _comparable_values(entity_type) <= every_type.get('', frozenset())
    fields = {**(every_type if known else {}), **scope.get(entity_type, {})} if entity_type else {}
    return all(name not in record or bool(_comparable_values(record[name]) & values) for name, values in fields.items())


def _world_unread_ids(pairs: Iterable[tuple[str, ThreadResult | None]], *, reader_scopes: ReaderScopes) -> dict[str, dict[str, Any]]:
    """The rows whose tools asked for a record the scenario SEEDED and came back with nothing.

    ``{scenario_id: {'calls': n, 'tools': [...]}}``. A call counts when it went through (no dispatch
    error, not a harness-gated tool), its response is a zero-row shape, and EVERY scalar value it
    was asked with is carried by ONE seeded record (its id included). That is "the arguments name a
    seeded entity" made literal: a legitimate no-match always asks with at least one value the
    world does not hold — a forged code, a date nothing was seeded on — so zero is its honest answer;
    a read that names one record exactly and finds nothing is a reader that cannot see the world it
    was handed (a mock filtering on one spelling while the seed carries another, a join on a key
    spelled differently).

    The client-side sibling of :func:`_degraded_ids`, for the same reason: ``/finalize`` grades a
    well-formed conversation over tools that answered, so nothing on that side can say the world was
    never read. Only this side holds the seeded world and every tool response together.

    NOT APPLICABLE by structure, with no branch of its own: an empty seeded world has no record for
    the arguments to belong to, and an unread tool channel (``tool_events`` ``None``) has no calls. A
    call with no argument values names no record either. Matched on field VALUES, never on entity-id
    shape or a type spelling — the arguments an agent sends are the values a user said.

    The stated false positive: a read naming one seeded record whose correct answer is an empty set
    of OTHER records (a seeded passenger who genuinely has no bookings). It is named in the channel
    reading; the demotion it would cost is a named channel with the exit code untouched.

    ``reader_scopes`` (:func:`_reader_scopes`) narrows the records a call is compared with to the ones its tool could
    return: a reservation-slot reader asked for a date that holds only delivery slots read nothing it could see, and
    zero is its honest answer. It narrows them by type too, to the types the
    tool's contract names: an order-line reader asked a product id that sits only on a catalog row read nothing it could see.
    So a match on a parent record (a passenger's name asked of a bookings reader) counts only when the contract names the
    parent type. The named risks: a contract that omits a type its reader touches, and read-type rows seeded only inline
    on another type, are not compared. An argument whose input names the field it is compared with
    (:data:`_ARG_TARGETS`) is compared with that field alone (:func:`_names_record`): the contract id a related-contracts
    listing filters parent pointers by names no row when no other contract points at it. Its named risks: a target the
    env-gen classifier judged wrong (it is checked only to be a field the tool reads; the same fact tells the mock's
    author which field to filter on), and a bool target whose ``true`` widens rather than narrows (``include_archived``;
    the backend's gloss lets only a narrowing ``true`` name one) hide a blind read of that argument. On a resumed run, rows the
    invocation did not re-run keep their prior answer (:func:`_reconcile_rows`).
    """
    unread: dict[str, dict[str, Any]] = {}
    for scenario_id, result in pairs:
        if result is None:
            continue
        # Shape-tolerant, because this runs after every scenario finished and before the summary is
        # written: a world or an event log this reader cannot parse contributes nothing, and must
        # never cost the run its run_summary.json.
        entities = (result.initial_world_state or {}).get('entities')
        items = entities.items() if isinstance(entities, dict) else ()
        seeded = [(record, _comparable_values(record) | _comparable_values(entity_id)) for entity_id, record in items]
        events = (result.tool_events or {}).get('events')
        tools = [
            str(event.get('tool') or '?')
            for event in (events if isinstance(events, list) else [])
            if _reads_a_seeded_record_as_empty(event, _readable_records(event, seeded, reader_scopes), reader_scopes)
        ]
        if tools:
            unread[scenario_id] = {'calls': len(tools), 'tools': sorted(set(tools))}
    return unread


def _readable_records(event: object, seeded: list[tuple[object, set[str]]], reader_scopes: ReaderScopes) -> list[tuple[object, set[str]]]:
    """The seeded records (with their values) a call to this event's tool could have returned (:func:`_in_reader_scope`)."""
    scope = reader_scopes.get(str(event.get('tool') or '')) if isinstance(event, dict) else None
    return [(record, values) for record, values in seeded if not scope or _in_reader_scope(record, scope)]


def _reads_a_seeded_record_as_empty(event: object, records: list[tuple[object, set[str]]], reader_scopes: ReaderScopes) -> bool:
    """One recorded event, against the seeded records: a call that went through, came back as a zero-row shape, and
    was asked ONLY with values one record carries (:func:`_names_record`)."""
    if not isinstance(event, dict) or event.get('type') != 'tool_call':
        return False
    if str(event.get('status') or '').lower() in _NOT_A_DISPATCHED_CALL or not _empty_read(event):
        return False
    args = event.get('args')
    scope = reader_scopes.get(str(event.get('tool') or ''), {})
    return bool(_comparable_values(args)) and any(_names_record(args, record, values, scope) for record, values in records)


def _names_record(args: object, record: object, values: set[str], scope: Mapping[str, Mapping[str, frozenset[str]]]) -> bool:
    """Whether ``record`` carries every asked value. An argument with a target (:data:`_ARG_TARGETS`) that names a field
    this record holds is compared with that field alone: a list asks for any of its values, a multi-field target
    matches on any field, and a bool narrows only when ``true`` (the field is non-empty); ``false``, ``None`` or an
    empty value narrow nothing. A value found on a target that is the record's own key (:data:`_KEY_TARGETS`) names
    the record whatever the other targeted REFERENCES ask (a listing that also fetches a given parent by id); a targeted
    filter that misses still clears it. Every untargeted argument's values must sit somewhere on the record.
    The named gap: two references a handler ANDs on one record, one of them a key hit, still fire."""
    if not isinstance(record, dict) or not isinstance(args, dict):
        return _comparable_values(args) <= values
    targets, keys, references = scope.get(_ARG_TARGETS, {}), scope.get(_KEY_TARGETS, {}), scope.get(_REFERENCE_ARGS, {})
    untargeted: list[object] = []
    missed_filter = missed_reference = keyed = False
    for arg, value in args.items():
        paths = [
            path
            for path in targets.get(str(arg), ())
            if path.partition('.')[2] in record and _comparable_values(path.partition('.')[0]) == _comparable_values(record.get('_entity_type'))
        ]
        if not paths:
            untargeted.append(value)
        elif isinstance(value, bool):
            missed_filter |= value and not any(record[path.partition('.')[2]] for path in paths)
        elif asked := _comparable_values(value):
            missed = not asked & _comparable_values([record[path.partition('.')[2]] for path in paths])
            if str(arg) in references:
                missed_reference |= missed
            else:
                missed_filter |= missed
            keyed |= bool(asked & _comparable_values([record[path.partition('.')[2]] for path in paths if path in keys.get(str(arg), ())]))
    return not missed_filter and (keyed or not missed_reference) and _comparable_values(untargeted) <= values


def _echoes_dropped_ids(pairs: Iterable[tuple[str, ThreadResult | None]]) -> dict[str, dict[str, int]]:
    """The rows whose AGENT channel was not simply what the adapter returned, counted BY RULE.

    ``{scenario_id: {'role_declared': n, 'verbatim_echo': n}}`` — the sibling of
    :func:`_degraded_ids`, and reported for the same reason: only this side knows. The backend
    grades whatever channel it was handed, so a run whose adapter returned the user's own words
    inside the agent's chain reads as perfectly clean everywhere downstream. A log line alone would
    not reach the operator: it is delivered at ``lastResort`` to stderr while every other
    progress line goes to stdout, so the routine ``| tee run.log`` loses it entirely.

    Counts, not messages: the summary is a verdict and the messages themselves are already in
    each transcript's ``metadata.user_echoes_dropped``.

    It does NOT ride ``measurement_clean``, unlike ``degraded_turns``, and the difference is the
    point: an emptied turn means a manufactured reply was SCORED, while a dropped echo means a
    foreign message was REMOVED before anything scored it. The transcript is right afterwards.
    What is left to report is the ADAPTER, which is what this map is.
    """
    return {
        scenario_id: {rule: len(msgs) for rule, msgs in result.user_echoes_dropped.items()}
        for scenario_id, result in pairs
        if result is not None and result.user_echoes_dropped
    }


def _graded_ids(cfg: _RunConfig, pairs: Iterable[tuple[str, ThreadResult | None]]) -> set[str]:
    """The ids whose scenario actually produced a policy verdict.

    Gated on ``cfg.is_scored`` AS WELL AS the evidence, deliberately: with no eval set
    attached there is nothing for the backend to grade, so a payload that somehow carried
    cells must not be able to manufacture a grading claim nobody asked for. Both must hold.
    """
    if not cfg.is_scored:
        return set()
    return {scenario_id for scenario_id, result in pairs if _scenario_was_graded(result)}


@dataclass(frozen=True, slots=True)
class _RunVerdict:
    """The ONE resolved answer both ``run_summary.json`` and the terminal report.

    It exists because deriving them separately lets them disagree: a file that recomputes
    its counts from the merged scenario map while ``_print_summary`` is handed this
    invocation's raw counters writes ``total: 7`` and prints ``1/1 scenarios succeeded`` in
    the same breath on a ``--scenario-id`` retry — and the merged ``scored`` floor, computed
    inside the writer's frame, never reaches the screen at all. Resolved ONCE here, returned,
    and printed from the same object.

    ``scenarios`` is the population the counts describe (merged across attempts on a
    partial run), and ``total``/``failures`` are derived FROM it — never copied from a
    caller's counters, which after a merge describe a subset.
    """

    #: The merged population this verdict describes, and the counts derived from it.
    scenarios: dict[str, str]
    total: int
    failures: int
    #: Rows that took no FULL measurement, in two shapes (see ``_row_outcome``): the per-scenario
    #: TIME budget ended them — they ran, landed with artifacts and were graded for the turns they
    #: reached — or the SUITE quarantined them and they never ran at all. Counted apart from both
    #: neighbours so a truncated or refused row is never reported as a clean one nor as a broken one.
    incomplete: int
    #: COUNTED, never derived as ``total - failures - incomplete``. A residual reports every
    #: row status it does not recognise as a success: the moment a fourth outcome exists, a
    #: subtraction puts it silently in the green column and out of the exit code. Counting
    #: instead makes an unrecognised row visibly belong to no bucket, which ``__post_init__``
    #: then refuses outright.
    succeeded: int
    #: The ids THIS invocation ran (``exit_code`` describes exactly these), and the scope tag.
    ran: list[str]

    @property
    def eval_revision(self) -> str | None:
        """THE revision that graded this run, when every graded row agrees; ``None`` when no row
        carries one (an older backend) or when rows disagree (a set edited mid-run — the per-row
        map says which)."""
        distinct = set(self.eval_revisions.values())
        if len(distinct) != 1:
            return None
        return distinct.pop()

    def __post_init__(self) -> None:
        # The three buckets PARTITION the population — stated in run_summary.json and read as a
        # partition by every consumer of it. A row status nobody counted would otherwise be
        # invisible: absent from all three numbers while still inflating ``total``.
        #
        # SAID, never raised. ``_merge_prior_scenarios`` deliberately keeps any string row it
        # finds in a prior run_summary.json (it validates shape, it does not coerce meaning), so
        # a file written by a future build can carry a status this one does not know. This
        # verdict is a report ABOUT a run; failing the run over an arithmetic surprise in its own
        # report would invert the classification the report exists to provide.
        counted = self.succeeded + self.failures + self.incomplete
        if counted != self.total:
            logger.warning(
                'run verdict buckets do not partition the population: %d counted (ok/failed/incomplete) vs %d rows. '
                'A row carries a status this build does not recognise — most likely a run_summary.json written by a '
                'newer runner into the same --out. Rows: %s',
                counted,
                self.total,
                self.scenarios,
            )

    exit_code_covers: str
    #: The rows that PRODUCED a measurement — the evidence every grading claim below is
    #: derived from, and the only one of them a merge can carry forward per row.
    graded: list[str]
    #: The rows whose dialog ended for something other than the user reaching their goal, mapped
    #: to the typed reason (``ThreadResult.ended_early``).
    #: A cut scenario scores low for obligations its conversation never reached, so a verdict
    #: that does not carry this reads that as the agent's failure.
    ended_early: dict[str, str]
    #: Whether EVERY row in ``scenarios`` was graded, and whether every row this invocation
    #: ran was. See :func:`_resolve_run_verdict`.
    scored: bool
    scored_this_invocation: bool
    #: Whether an ``--eval-set-id`` was attached at all. NOT a grading claim — it is the CLI
    #: fact ``scored`` is easily mistaken for, kept apart so the banner can tell "you asked
    #: for no grading" from "you asked for grading and none landed".
    eval_set_attached: bool
    #: Per row whose simulated user stated WHY it gave up, that cause
    #: (``ThreadResult.ended_early_cause``). ``ended_early`` says WHO stopped; this says what it
    #: saw, and only this separates "the agent refused" from "the user never saw the reply".
    ended_early_causes: dict[str, str] = dataclasses_field(default_factory=dict)
    #: The cut rows carried from a prior attempt whose cause NO WRITER RECORDED (a summary with no
    #: ``ended_early_causes`` key). Their absence from the map above is not the backend
    #: stating none, and this is the one place that says so.
    ended_early_causes_unrecorded: list[str] = dataclasses_field(default_factory=list)
    #: Per row, the backend's ``measurement_integrity`` block (rows that sent none are absent).
    measurement_integrity: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per row, the backend's ``user_sim_integrity`` block — the SCENARIO-fidelity half of the
    #: apparatus (:func:`_user_sim_integrity`). Kept apart from the block above because the two
    #: have different owners: a broken judge is an eval fix, a simulator that skipped a scheduled
    #: question is not repairable by any eval change. They meet only in ``measurement_clean``.
    user_sim_integrity: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per row, the eval set revision that graded it (``_eval_revisions``). Rows an older backend
    #: scored are absent.
    eval_revisions: dict[str, str] = dataclasses_field(default_factory=dict)
    #: Per row, how many turns the HARNESS emptied (:func:`_degraded_ids`). The CLIENT-side half
    #: of measurement integrity: the backend scored a well-formed conversation and says so, and
    #: only the runner knows some of those replies were manufactured here.
    degraded: dict[str, int] = dataclasses_field(default_factory=dict)
    #: Per row, the tool calls that asked for ONE seeded record and came back empty
    #: (:func:`_world_unread_ids`): ``{calls, tools}``. The other client-side-only breach — the
    #: backend graded answers from tools that never read the world this scenario was seeded with.
    world_unread: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per row whose tool channel went unread on a bench declaring ``agent.native_tools``, those
    #: tools (:func:`_note_tool_channel`). Not a breach: a channel nobody measured.
    tool_channel_unread: dict[str, list[str]] = dataclasses_field(default_factory=dict)
    #: Per row whose agent was served a stub for a tool call the mock never answered
    #: (:func:`_note_mock_unanswered`): ``{calls, restart_shaped, served_schema_stub}``. A breach of
    #: the measurement — some tool answers the backend graded were stubs — never of the exit code.
    mock_unanswered: dict[str, dict[str, int]] = dataclasses_field(default_factory=dict)
    #: Per row whose child claimed nothing about unanswered calls, ``True`` (:func:`_note_mock_unanswered`).
    #: Not a breach: a channel nobody measured (``measurement_unchecked_because: mock_unanswered``).
    mock_unanswered_unclaimed: dict[str, bool] = dataclasses_field(default_factory=dict)
    #: Per row, how many replies the child trimmed to fit one IPC line (:func:`_note_ipc_trimmed`). A breach of the
    #: measurement — the judges graded those turns without their message chain — never of the exit code.
    ipc_trimmed: dict[str, int] = dataclasses_field(default_factory=dict)
    #: Per row, how many messages the single-role split took out of the AGENT channel, by rule
    #: (:func:`_echoes_dropped_ids`). The other client-side-only fact about the channel a judge
    #: reads.
    echoes_dropped: dict[str, dict[str, int]] = dataclasses_field(default_factory=dict)
    #: Per row, the provenance of a ceiling-hit re-run (:class:`_TurnCapRerunProvenance` plus
    #: whether the attempt it superseded had been graded). Present only under
    #: ``--rerun-on-turn-cap``.
    turn_cap_reruns: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per row, ``{started_at, ended_at, duration_seconds}`` for the WHOLE row (every re-drive
    #: and ceiling re-run inside it). It is the run's only recorded clock; artifact file mtimes are not.
    timings: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per row that earned one, the STABILITY retry's record: why it was retried,
    #: whether the failure reproduced, which attempt is authoritative, and the eval revision each
    #: attempt was graded at (with ``marks_valid: false`` where they disagree). Present only for
    #: rows in doubt — a missing row means the scenario was never in doubt, never that it was
    #: stable.
    stability: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: Per declared family, the pass-rate gradient (:func:`_family_gradient`). ``0/7`` says only
    #: "everything failed"; 0% in one family and 100% elsewhere is a defect with an address.
    family_gradient: dict[str, Any] = dataclasses_field(default_factory=dict)
    #: THREE-VALUED, and stated rather than derived from an empty map. ``True`` — EVERY row in
    #: ``scenarios`` asserted a block and none reported a breach. ``False`` — some row reported
    #: one. ``None`` — measurement integrity is UNKNOWN for this run: no row carried a block (an
    #: older backend), or only some rows did, which answers for those rows and says nothing
    #: about the rest. A two-valued flag would have to call both of those clean, which is the
    #: absence-reads-as-good-news shape the block exists to remove. See :func:`_measurement_clean`.
    measurement_clean: bool | None = None
    #: WHICH channels drove a ``False`` above (:func:`_measurement_unclean_because`). Empty on
    #: every other verdict — a reason list beside ``True`` or ``None`` would be a claim about a
    #: breach nobody found.
    #:
    #: NOT independently readable, and deliberately two-valued where its neighbour is three: ``[]``
    #: covers clean, unknown, AND a ``False`` carried without its reasons. Read it BESIDE
    #: ``measurement_clean``, never instead of it — ``if not because: clean = True`` is the
    #: absence-reads-as-good-news inference the tri-state flag exists to prevent.
    measurement_unclean_because: list[str] = dataclasses_field(default_factory=list)
    #: WHICH channels were NOT MEASURED on some row (:func:`_measurement_unchecked_because`) —
    #: the reason a ``None`` above is a coverage gap rather than an older backend. Empty beside
    #: ``False`` says nothing: a breach elsewhere decides the verdict whatever was unmeasured.
    measurement_unchecked_because: list[str] = dataclasses_field(default_factory=list)
    #: Per row that returned a result, the house verdict status (:func:`_scenario_verdict_ids`) —
    #: what the headline tallies. Row status says the thread COMPLETED; this says whether it PASSED.
    scenario_verdicts: dict[str, str] = dataclasses_field(default_factory=dict)
    #: Per row whose verdict served one, the backend's ``high_severity_violations`` — the headline's severity axis.
    scenario_high_severity: dict[str, list[str]] = dataclasses_field(default_factory=dict)
    #: Per ``plan_exhausted`` row that ended on a push-back, the agent turn it disputed (:func:`_plan_exhausted_disputed_turns`).
    plan_exhausted_disputed_turns: dict[str, int] = dataclasses_field(default_factory=dict)
    #: Per row the agent closed before its target beat arose, that beat (:func:`_agent_close_preempted_targets`).
    agent_close_preempted_targets: dict[str, str] = dataclasses_field(default_factory=dict)
    #: Per row that ended early, its ONE route ``{owner, remedy, qualifier?}`` (:func:`_resolve_ending`) —
    #: what the console prints and the run report renders.
    ended_early_routes: dict[str, dict[str, str]] = dataclasses_field(default_factory=dict)
    #: A partial run's prior ``host_effects`` block (:attr:`_PriorRun.host_effects`); empty on a full run.
    prior_host_effects: dict[str, Any] = dataclasses_field(default_factory=dict)


def _measurement_clean(
    integrity: dict[str, Any],
    population: dict[str, str],
    degraded: dict[str, int] | None = None,
    user_sim: dict[str, Any] | None = None,
    world_unread: dict[str, Any] | None = None,
    tool_channel_unread: Mapping[str, Any] | None = None,
    mock_unanswered: Mapping[str, Any] | None = None,
    mock_unanswered_unclaimed: Mapping[str, Any] | None = None,
    ipc_trimmed: Mapping[str, Any] | None = None,
    *,
    graded: Collection[str],
) -> bool | None:
    """The run-level measurement-integrity verdict: ``True`` / ``False`` / ``None``.

    ``False`` needs ONE breach — a row that reported one is a fact about the run whatever the
    other rows did or did not say. FOUR channels can report one, and this function is the only
    place all four meet:

    * the backend's own ``measurement_integrity`` block (an errored judge, a raising
      applicability guard, a crashed per-turn pass, a broken deterministic target check);
    * HARNESS-EMPTIED turns (``degraded``), the breach the backend structurally cannot report:
      ``/finalize`` graded a well-formed conversation and its block says ``clean: true``, while
      some of the replies it graded were manufactured by the runner rather than given by the
      agent;
    * the backend's ``user_sim_integrity`` block (``user_sim``) — the simulated USER departing
      from the scenario it was given;
    * WORLD-UNREAD reads (``world_unread``, :func:`_world_unread_ids`), the second breach only the
      client can see: tools asked for a record the scenario seeded and came back empty, so the
      backend graded a well-formed conversation over an environment that never read its own world;
    * MOCK-UNANSWERED calls (``mock_unanswered``, :func:`_note_mock_unanswered`): the mock never
      answered some tool calls, so the agent read the runner's stubs and the backend graded a dialog
      in which those tool answers were not the mock's.

    Including it is a claim about SCOPE, not a merging of two concerns: an unfaithfully-driven
    dialog is a conversation nobody authored, so what was scored on it was not the scenario. The
    per-row maps stay separate in ``run_summary.json`` precisely because the remedies differ —
    only this one boolean speaks for the run as a whole.

    ``True`` needs EVERY ``graded`` row to have asserted a ``measurement_integrity`` block. A block
    from one row of three answers for that row and says nothing about the other two, so reporting
    the run as clean on it is the same absence-reads-as-good-news the three-valued field exists
    to prevent: the two silent rows are exactly the rows whose measurement may have
    died. Coverage is keyed on that block alone — a backend old
    enough to send no user-sim block at all must not make every run permanently UNKNOWN; a
    user-sim block can only ever turn a verdict to ``False``. A row that was not graded (never
    reached ``/finalize``, or was refused there) measured nothing and is outside that population:
    its failure is already typed (``failed`` / ``incomplete``, ``untrustworthy_scenarios``) and
    ``scored: false``, and the breach channels still read it.

    Everything else is ``None`` — unknown. That covers both "no row carried a block" (an older
    backend) and partial coverage; ``measurement_coverage`` in ``run_summary.json`` is the
    denominator that tells a reader which of the two it is.

    A user-sim row the backend marked UNCHECKED is partial coverage too, not a breach. The block
    publishes a three-valued ``state`` beside ``clean`` (``'anomalies' | 'unchecked' | 'clean'``),
    and ``clean: false`` with ``state: unchecked`` says the evaluator-voice detector could not
    READ some turns (its patterns are English; the persona switched language) — nothing departed,
    nothing was cleared. Folding that on ``clean`` alone would report every bilingual scenario as
    an unfaithfully driven dialog. Such a row makes
    the verdict ``None`` and lands in :func:`_measurement_unchecked_because`.
    """
    if _measurement_unclean_because(integrity, population, degraded, user_sim, world_unread, mock_unanswered, ipc_trimmed):
        return False
    if not integrity or not set(graded) <= set(integrity):
        return None
    if _measurement_unchecked_because(population, user_sim, tool_channel_unread, mock_unanswered_unclaimed):
        return None
    return True


def _user_sim_unchecked(block: Mapping[str, Any]) -> bool:
    """Whether a ``user_sim_integrity`` row says NOT MEASURED rather than clean or breached.

    Only a producer that publishes ``state`` can say so; a block with a bare ``clean: false``
    stays a breach.
    """
    return block.get('state') == 'unchecked'


def _user_sim_breached(block: Mapping[str, Any]) -> bool:
    """Whether a ``user_sim_integrity`` row reported a DEPARTURE — unclean, and not merely unchecked."""
    return block.get('clean') is not True and not _user_sim_unchecked(block)


def _measurement_unchecked_because(
    population: dict[str, str],
    user_sim: dict[str, Any] | None = None,
    tool_channel_unread: Mapping[str, Any] | None = None,
    mock_unanswered_unclaimed: Mapping[str, Any] | None = None,
) -> list[str]:
    """WHICH channels could not be measured on some row — why :func:`_measurement_clean` is ``None``
    when every row did assert a block. ``tool_channel``: a row's declared native tools ran where no
    lane recorded them (:func:`_note_tool_channel`). ``mock_unanswered``: a row's child claimed nothing
    about which tool calls the mock left unanswered (:func:`_note_mock_unanswered`).

    Beside ``measurement_unclean_because``, never folded into it: an unmeasured channel names a
    coverage gap (read the turns), a breached one names a defect (fix the scenario), and the
    remedies differ.
    """
    channels: list[str] = []
    if any(sid in population and _user_sim_unchecked(block) for sid, block in (user_sim or {}).items()):
        channels.append('user_sim_integrity')
    if any(sid in population for sid in (tool_channel_unread or {})):
        channels.append('tool_channel')
    if any(sid in population for sid in (mock_unanswered_unclaimed or {})):
        channels.append('mock_unanswered')
    return channels


def _measurement_unclean_because(
    integrity: dict[str, Any],
    population: dict[str, str],
    degraded: dict[str, int] | None = None,
    user_sim: dict[str, Any] | None = None,
    world_unread: dict[str, Any] | None = None,
    mock_unanswered: Mapping[str, Any] | None = None,
    ipc_trimmed: Mapping[str, Any] | None = None,
) -> list[str]:
    """WHICH channels reported a breach — the reason :func:`_measurement_clean` is ``False``.

    The verdict alone is unreadable at the top of a summary: a reader sees ``measurement_clean:
    false`` beside a ``measurement_integrity`` map in which every row says ``clean: true``, and
    nothing in the file says it was the user-sim block next door that decided it. That reads as a
    contradiction in the artifact rather than as the finding it is.

    Named channels, not prose: each one routes to a different owner (the harness, the judges,
    the simulated user, the world reader), which is the same reason the four maps are kept apart
    on disk.
    """
    channels: list[str] = []
    if any(sid in population for sid in (degraded or {})):
        channels.append('degraded_scenarios')
    if any(block.get('clean') is not True for block in integrity.values()):
        channels.append('measurement_integrity')
    if any(sid in population and _user_sim_breached(block) for sid, block in (user_sim or {}).items()):
        channels.append('user_sim_integrity')
    if any(sid in population for sid in (world_unread or {})):
        channels.append('world_unread_scenarios')
    if any(sid in population for sid in (mock_unanswered or {})):
        channels.append('mock_unanswered_scenarios')
    if any(sid in population for sid in (ipc_trimmed or {})):
        channels.append('ipc_trimmed_scenarios')
    return channels


def _rows_in[RowFact](rows: Mapping[str, RowFact] | None, population: Container[str], *, excluding: Container[str] = ()) -> dict[str, RowFact]:
    """The per-row facts whose scenario id is in ``population`` and not in ``excluding``.

    ONE scoping rule, obeyed by every per-row map the verdict carries (ended-early reasons,
    integrity blocks, degradation counts, ceiling re-runs). Written once because they must not
    diverge: a map that quietly kept a row the others dropped would let one claim outlive the
    evidence for it.
    """
    return {sid: value for sid, value in (rows or {}).items() if sid in population and sid not in excluding}


def _reconcile_rows[RowFact](
    this_run: Mapping[str, RowFact] | None,
    carried: Mapping[str, RowFact],
    *,
    ran: Container[str],
    merged: Container[str],
) -> dict[str, RowFact]:
    """One per-row map, scoped to the merged population and answered by the right attempt.

    Rows this invocation RAN are answered by this invocation's evidence; rows it did not run
    keep the PRIOR attempt's answer. That rule is the SAME for every per-row map, including the
    two client-side ones — a re-run that came back clean must LOSE the earlier row's degradation
    count rather than inherit it, or a repaired adapter stays permanently suspect — so it is
    written once here rather than seven times at the call site, where the seventh copy is the one
    that gets it wrong.

    The closing scope to ``merged`` is deliberate belt-and-braces: ``ran`` is a subset of
    ``merged`` by construction (:func:`_merge_prior_scenarios` overlays it last), so this drops
    nothing today and is what keeps a map from outliving the population if that ever changes.
    """
    return _rows_in(_rows_in(this_run, ran) | _rows_in(carried, merged, excluding=ran), merged)


def _resolve_run_verdict(
    out: Path,
    *,
    scenarios: dict[str, str],
    graded: set[str],
    cfg: _RunConfig,
    ended_early: dict[str, str] | None = None,
    ended_early_causes: dict[str, str] | None = None,
    measurement_integrity: dict[str, Any] | None = None,
    user_sim_integrity: dict[str, Any] | None = None,
    eval_revisions: dict[str, str] | None = None,
    degraded: dict[str, int] | None = None,
    echoes_dropped: dict[str, dict[str, int]] | None = None,
    turn_cap_reruns: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
    family_gradient: dict[str, Any] | None = None,
    world_unread: dict[str, Any] | None = None,
    tool_channel_unread: dict[str, list[str]] | None = None,
    mock_unanswered: dict[str, dict[str, int]] | None = None,
    mock_unanswered_unclaimed: dict[str, bool] | None = None,
    ipc_trimmed: dict[str, int] | None = None,
    scenario_verdicts: dict[str, str] | None = None,
    scenario_high_severity: dict[str, list[str]] | None = None,
    plan_exhausted_disputed_turns: dict[str, int] | None = None,
    agent_close_preempted_targets: dict[str, str] | None = None,
) -> _RunVerdict:
    """Reconcile this invocation's outcomes with any prior attempt's, into one verdict.

    **Which population the counts describe depends on ``cfg.is_partial``**, and the verdict
    names the scope rather than leaving a reader to infer it. A full run describes only
    itself. A ``--scenario-id`` run MERGES the prior summary, so its counts span several
    attempts while ``exit_code`` still describes only ``ran``  — the two can legitimately
    disagree (``failed: 3`` beside ``exit_code: 0`` after a retry fixed the last one), and
    ``exit_code_covers`` names the scope so nothing has to guess.

    The counts are recomputed from the merged map UNCONDITIONALLY: recomputing them only on the
    partial branch would let the full/experiment path report a caller's counters beside a map
    that disagrees with them (the vendor sink reports a failure COUNT, the map records
    per-scenario outcomes, and they come from different places).

    ``graded`` is the EVIDENCE — the ids this invocation watched produce a policy verdict
    (:func:`_scenario_was_graded`). Every grading claim in the verdict is derived from it and
    from the prior attempt's own recorded ids; none is derived from ``--eval-set-id``, which
    is a statement about a command line and not about anything that was measured.
    """
    ran = dict(scenarios)
    # A FULL run reads no prior file — its population and its per-row answers are its own. That
    # is said as an empty prior rather than as a branch around every map below, so the merge rule
    # has exactly one implementation and the full path cannot drift out of it.
    prior = _merge_prior_scenarios(out, scenarios) if cfg.is_partial else _PriorRun(scenarios=dict(scenarios))
    merged = prior.scenarios
    # Only the rows this invocation RAN may be answered by this invocation's evidence; the rest
    # keep what the prior attempt recorded (see :func:`_reconcile_rows`).
    graded_ids = ({sid for sid in graded if sid in ran} | {sid for sid in prior.graded if sid in merged and sid not in ran}) & set(merged)
    early = _reconcile_rows(ended_early, prior.ended_early, ran=ran, merged=merged)
    causes = _reconcile_rows(ended_early_causes, prior.ended_early_causes, ran=ran, merged=merged)
    # Only rows STILL carried from the prior: a row this invocation re-ran was asked afresh.
    unrecorded = sorted(sid for sid in prior.ended_early_causes_unrecorded if sid in early and sid not in ran)
    integrity = _reconcile_rows(measurement_integrity, prior.measurement_integrity, ran=ran, merged=merged)
    user_sim = _reconcile_rows(user_sim_integrity, prior.user_sim_integrity, ran=ran, merged=merged)
    disputed = _reconcile_rows(plan_exhausted_disputed_turns, prior.plan_exhausted_disputed_turns, ran=ran, merged=merged)
    preempted = _reconcile_rows(agent_close_preempted_targets, prior.agent_close_preempted_targets, ran=ran, merged=merged)
    harmed = _reconcile_rows(degraded, prior.degraded, ran=ran, merged=merged)
    unread = _reconcile_rows(world_unread, prior.world_unread, ran=ran, merged=merged)
    tools_unread = _reconcile_rows(tool_channel_unread, prior.tool_channel_unread, ran=ran, merged=merged)
    unanswered = _reconcile_rows(mock_unanswered, prior.mock_unanswered, ran=ran, merged=merged)
    unclaimed = _reconcile_rows(mock_unanswered_unclaimed, prior.mock_unanswered_unclaimed, ran=ran, merged=merged)
    trimmed = _reconcile_rows(ipc_trimmed, prior.ipc_trimmed, ran=ran, merged=merged)
    echoes = _reconcile_rows(echoes_dropped, prior.echoes_dropped, ran=ran, merged=merged)
    reruns = _reconcile_rows(turn_cap_reruns, prior.turn_cap_reruns, ran=ran, merged=merged)
    clocks = _reconcile_rows(timings, prior.timings, ran=ran, merged=merged)
    # Reconciled like every other per-row map, so a --scenario-id retry keeps what an earlier
    # attempt recorded for the rows it did not re-run.
    stability_rows = _reconcile_rows(stability, prior.stability, ran=ran, merged=merged)
    return _RunVerdict(
        scenarios=merged,
        total=len(merged),
        # ``== 'failed'``, never ``!= 'ok'``: an ``incomplete`` row is neither, and counting it
        # as a failure would fail a run whose scenarios all landed and were graded.
        failures=sum(1 for v in merged.values() if v == 'failed'),
        incomplete=sum(1 for v in merged.values() if v == 'incomplete'),
        succeeded=sum(1 for v in merged.values() if v == 'ok'),
        ran=sorted(ran),
        exit_code_covers='ran_this_invocation' if cfg.is_partial else 'all',
        graded=sorted(graded_ids),
        ended_early=dict(sorted(early.items())),
        ended_early_causes=dict(sorted(causes.items())),
        ended_early_causes_unrecorded=unrecorded,
        # Non-empty populations only: an empty map has no row that went ungraded, and
        # ``set() >= set()`` would otherwise report a run that measured nothing as fully
        # measured.
        scored=bool(merged) and graded_ids >= set(merged),
        scored_this_invocation=bool(ran) and set(ran) <= graded_ids,
        eval_set_attached=cfg.is_scored,
        measurement_integrity=dict(sorted(integrity.items())),
        user_sim_integrity=dict(sorted(user_sim.items())),
        eval_revisions=dict(sorted((eval_revisions or {}).items())),
        degraded=dict(sorted(harmed.items())),
        world_unread=dict(sorted(unread.items())),
        tool_channel_unread=dict(sorted(tools_unread.items())),
        mock_unanswered=dict(sorted(unanswered.items())),
        mock_unanswered_unclaimed=dict(sorted(unclaimed.items())),
        ipc_trimmed=dict(sorted(trimmed.items())),
        echoes_dropped=dict(sorted(echoes.items())),
        turn_cap_reruns=dict(sorted(reruns.items())),
        timings=dict(sorted(clocks.items())),
        stability=dict(sorted(stability_rows.items())),
        # NOT reconciled: a gradient is an aggregate over the rows THIS invocation holds
        # payloads for, and merging one computed over a different population would report a
        # rate nobody measured. A partial run's gradient describes its own subset, and the
        # ``coverage`` inside it says how many rows that was.
        #
        # Defaulted to the EMPTY GRADIENT rather than ``{}``: the early-exit writer supplies
        # nothing, and a consumer reading ``['coverage']['reported']`` would meet a KeyError on
        # exactly the runs a guard refused — while ``.get('families', {})`` would read "computed,
        # found none" instead of "this writer recorded none". One shape, always.
        family_gradient=dict(family_gradient) if family_gradient else _family_gradient([]),
        # Against the MERGED population, not against the blocks alone: coverage is half the
        # verdict, and a row that sent no block is a row nothing was asserted about. The
        # harness-emptied rows ride in because they are the breach only this side can see, and
        # the user-sim rows because a dialog driven off its own scenario was not a measurement
        # of that scenario at all, and the world-unread rows because tools blind to the seeded
        # world answered for an environment the scenario never had.
        measurement_clean=_measurement_clean(integrity, merged, harmed, user_sim, unread, tools_unread, unanswered, unclaimed, trimmed, graded=graded_ids),
        measurement_unclean_because=_measurement_unclean_because(integrity, merged, harmed, user_sim, unread, unanswered, trimmed),
        measurement_unchecked_because=_measurement_unchecked_because(merged, user_sim, tools_unread, unclaimed),
        scenario_verdicts=dict(sorted(_reconcile_rows(scenario_verdicts, prior.scenario_verdicts, ran=ran, merged=merged).items())),
        scenario_high_severity=dict(sorted(_reconcile_rows(scenario_high_severity, prior.scenario_high_severity, ran=ran, merged=merged).items())),
        plan_exhausted_disputed_turns=dict(sorted(disputed.items())),
        agent_close_preempted_targets=dict(sorted(preempted.items())),
        # Resolved from the RECONCILED maps, so a partial re-run routes the rows it carried the same way.
        ended_early_routes={
            sid: _resolve_ending(
                reason,
                cause=causes.get(sid),
                disputed_turn=disputed.get(sid),
                breaches=(integrity.get(sid) or {}).get('breaches') or (),
                preempted_target=preempted.get(sid),
            )
            for sid, reason in sorted(early.items())
        },
        prior_host_effects=prior.host_effects,
    )


#: What this host gives the agent of what its own files declare it needs, resolved before the run
#: (:func:`_preflight_host_toolchain`) — the same module-level handoff as ``_LAST_WORLD_RECORD`` below,
#: for the same reason. Empty means no rung asked (an embedder, or a refusal above this one).
_LAST_HOST_TOOLCHAIN_RECORD: dict[str, Any] = {}


def _child_python() -> str:
    """The interpreter the agent's child runs in (``PLURAI_CHILD_PYTHON``, else this one) — ``_subprocess``'s own rule."""
    return os.environ.get('PLURAI_CHILD_PYTHON') or sys.executable


def _preflight_host_toolchain(cfg: _RunConfig) -> None:
    """Report, before any spend, what this host does not give the agent of what its own files declare it needs.

    Every ``harness.host_requirements`` entry is resolved where the agent will use it, in the AGENT's
    interpreter (:mod:`plurai.host_toolchain`, the one resolver the plugin's run-gate calls too):
    ``present``, ``missing``, ``fails_sandboxed``, ``broken``, ``unverified`` (no probe) or
    ``not_checked``, each with its remedy, and one install block for what is missing while its gap
    still holds (:func:`plurai.host_toolchain.install_block`; printed, never recorded). Each non-present one is a gate finding
    (:func:`plurai.host_toolchain.host_findings`): a measured gap holds the run at the funnel until an
    ``--overrule`` lifts it, and an unmeasured one ships with its finding. A resolver that cannot run,
    or does not finish, is ``not_checked`` for every declared name. Recorded for ``run_summary.json``
    whatever the verdict — the measurement only, never who lifted what.
    """
    _LAST_HOST_TOOLCHAIN_RECORD.clear()
    workdir = cfg.agent_module_path.parent
    declared = declared_host_requirements(workdir)
    if declared == ():
        _LAST_HOST_TOOLCHAIN_RECORD.update({'requirements': [], 'stamp': None})
        return
    argv = [_child_python(), '-m', 'plurai.host_toolchain', '--workdir', str(workdir), '--json']
    stderr_tail = ''
    # Each probe has its own budget inside the resolver; this bounds the resolver itself.
    budget = len(declared or ()) * HOST_PROBE_TIMEOUT_SECONDS + 60
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=budget, **utf8_child_text())  # noqa: S603 — the agent's own interpreter, a fixed module
        stderr_tail = ' '.join(done.stderr.split())[-300:]
        toolchain = json.loads(done.stdout.strip().splitlines()[-1])
        if not isinstance(toolchain, dict) or not isinstance(toolchain.get('requirements'), list):
            raise ValueError('no requirements list')
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        why = f"the resolver could not run in the agent's interpreter ({type(exc).__name__}: {exc}{'; ' + stderr_tail if stderr_tail else ''})"
        names = [(r.name, r.kind) for r in declared] if declared else [('host_requirements', 'declaration')]
        toolchain = {'confinement': None, 'requirements': [{'name': n, 'kind': k, 'status': 'not_checked', 'detail': why} for n, k in names]}
    # the record carries the measurement; the install block (it names a local folder) is printed, never recorded
    _LAST_HOST_TOOLCHAIN_RECORD.update({**{k: v for k, v in toolchain.items() if k != 'install'}, 'stamp': host_stamp(toolchain) or None})
    gates = current_gate_authority()
    for finding in host_findings(toolchain, declared or ()):
        gates.emit(finding)
    # ONE copy-paste block for what is missing AND still holds the run (an overruled gap asks the operator for
    # nothing), built only from the agent's own cited install lines; the operator runs it (host writes are theirs),
    # and the funnel below stops the run.
    holding = {f.subject for f in gates.held() if f.gate_id == HOST_REQUIREMENT_GAP}
    rows = [r for r in toolchain['requirements'] if r.get('name') in holding]
    if block := install_block(rows, declared or (), workdir.resolve().parent):
        print(block, file=sys.stderr)


#: What this run could READ of the world it scores, so ``run_summary.json`` can carry it — the same
#: module-level handoff ``_LAST_EGRESS_RECORD`` uses and for the same reason: the declaration is
#: answered in ``main()`` before either strategy and the summary is written after them, with no
#: frame spanning both. Empty means no rung asked (an embedder, or a refusal above this one).
_LAST_WORLD_RECORD: dict[str, Any] = {}
#: The ``shared_world`` named risk this run carries (:func:`plurai.orchestrator.orchestrator.shared_world_risk`),
#: empty when it carries none. Written beside ``_LAST_WORLD_RECORD`` by the same rung and read only while
#: that record is non-empty, so a writer no rung reached says "nobody asked" for both.
_LAST_SHARED_WORLD_RISK: dict[str, Any] = {}


def _shared_world_summary() -> dict[str, Any]:
    """``run_summary.json``'s ``named_risks`` + ``stability_comparable`` (+ its reason), off the world rung.

    ``named_risks`` is the run-level carrier the plugin's release verdict reads beside the backend's
    ``release_gate.named_risks`` (same entry keys). ``stability_comparable`` is ``False`` with reason
    ``shared_world`` when conversations share one unreset world (a retry or another trial ran on other
    writes, so their agreement is not a measurement of the agent), ``True`` when the rung ran and found
    none, and ``None`` when no rung asked — three different runs, never one.
    """
    if not _LAST_WORLD_RECORD:
        return {'named_risks': [], 'stability_comparable': None, 'stability_comparable_reason': None}
    risk = dict(_LAST_SHARED_WORLD_RISK)
    return {
        'named_risks': [risk] if risk else [],
        'stability_comparable': not risk,
        'stability_comparable_reason': SHARED_WORLD_RISK_KIND if risk else None,
    }


def _record_world_observability(simulate_path: Path, *, reason: str | None, slots: int | None = None, shared_world: dict[str, Any] | None = None) -> None:
    """Record WHICH world this run scores and whether anything here can read it.

    The consequence is already realised — a world the mock does not own and the harness does not
    hand over reaches the child as ``None`` and the backend demotes every world-reading policy to
    not-measured — and without this record the CAUSE reaches no artifact at all: one stderr
    ``WARNING:`` on a stream the routine ``| tee run.log`` keeps but no reader of the run parses.
    A report, a critic sub-agent or ``score_run`` then sees N world-reading policies at
    not-measured with no reason on file and attributes it to the backend or to the eval set.

    WRITTEN FOR EVERY RUN, warned or not, on the rule its neighbours in the summary state:
    "it was established", "it was not" and "nobody asked" are three different runs, and an absent
    key makes the third read like the first. So a mock-owned run records ``reason: None`` rather
    than nothing at all.

    ``snapshot_hook`` is read off the same module-scope walk the gates use and is meaningful for
    every owner, because the child dispatches the world channel on the HOOK and reads the owner
    only to decide what its absence means (``child._ChildSession.handle_finalize``): a mock-owned
    bench that exports it has asked for its own answer and gets it. A harness nothing could parse
    answers ``False`` here; beside a non-mock owner that case reaches this line only once the
    holding finding of :func:`unreset_external_world_finding` one rung up has been overruled.

    ``owner`` is the EFFECTIVE owner, so a bench that declares nothing records ``mock`` rather
    than a ``null`` every reader would have to know the rule to resolve: ``_declared_world_owner``
    states that an undeclared world IS the mock's, and the child, both gates and this record must
    not become four readings of one word.

    ``slots`` is the EFFECTIVE number of world copies on the same rule: the declared
    ``world_state.slots``, or 1 when undeclared (one world, run one attempt at a time if external).
    """
    declared = _declared_world_owner(simulate_path)
    bound = _module_scope_bindings_of_file(simulate_path)
    _LAST_WORLD_RECORD.clear()
    _LAST_WORLD_RECORD.update(
        {
            'owner': declared[0] if declared is not None else WORLD_OWNER_MOCK,
            'snapshot_hook': bound is not None and WORLD_SNAPSHOT_HOOK in bound,
            'reason': reason,
            'slots': slots if slots is not None else 1,
        }
    )
    _LAST_SHARED_WORLD_RISK.clear()
    _LAST_SHARED_WORLD_RISK.update(shared_world or {})


def _write_run_summary(
    out: Path,
    *,
    exit_code: int,
    scenarios: dict[str, str],
    graded: set[str],
    cfg: _RunConfig,
    run_name: str | None = None,
    ended_early: dict[str, str] | None = None,
    ended_early_causes: dict[str, str] | None = None,
    measurement_integrity: dict[str, Any] | None = None,
    user_sim_integrity: dict[str, Any] | None = None,
    eval_revisions: dict[str, str] | None = None,
    degraded: dict[str, int] | None = None,
    echoes_dropped: dict[str, dict[str, int]] | None = None,
    turn_cap_reruns: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    stability: dict[str, Any] | None = None,
    family_gradient: dict[str, Any] | None = None,
    world_unread: dict[str, Any] | None = None,
    scenario_verdicts: dict[str, str] | None = None,
    scenario_high_severity: dict[str, list[str]] | None = None,
    plan_exhausted_disputed_turns: dict[str, int] | None = None,
    agent_close_preempted_targets: dict[str, str] | None = None,
    egress_dir: Path | None = None,
    aborted: Mapping[str, Any] | None = None,
    not_run: Sequence[str] = (),
) -> _RunVerdict:
    """Write ``<out>/run_summary.json`` — the run's own machine-readable verdict — and RETURN it.

    ``aborted`` is the ``to_record()`` of the run's abort; it is written as the ``aborted`` key (``null`` otherwise). A run
    stopped because a mock could not answer (:class:`_MockAbort`) passes no rows, so the file carries no verdict of this
    invocation. A run stopped for credit (:class:`_CreditAbort`) passes the rows it has: the finished scenarios keep their
    verdicts and ``not_run`` names the ones the stop kept from finishing, which are ``incomplete`` rows.

    The return is what lets the terminal report the same numbers the file does: the resolved
    verdict is computed BEFORE the write, so a best-effort write failure cannot leave the
    caller without one.

    The grading keys, and what each of them may be trusted to mean:

    * ``graded_scenarios`` — the rows that PRODUCED a policy verdict. The measurement itself,
      per row, and the only key a merge can carry forward honestly.
    * ``scored`` — EVERY row in ``scenarios`` is in ``graded_scenarios``. A scenario that was
      never graded still "succeeds" (it ran to completion), so succeeded/failed alone read
      exactly like a fully graded run while every score in the artifacts is a 0.0 meaning
      "not measured". This is a DERIVED claim about evidence, never ``bool(eval_set_id)``:
      that would state a command-line argument no part of the run checks, and an empty eval
      set, a suite of errored judges, or a ``status: 'failed'`` finalize each falsify it
      while the file still says ``true``.
    * ``scored_this_invocation`` — the same claim narrowed to the ids this invocation ran.
    * ``eval_set_attached`` — the CLI fact, kept under its own honest name so nothing has to
      reach for ``scored`` to answer "was an eval set even passed".

    ``graded`` is required, not defaulted: a caller that cannot say what was measured must be
    unable to publish a grading claim at all.

    ``measurement_integrity`` carries the backend's own per-row block. It is the answer to a
    question none of the keys above can reach: was anything on this row left UNMEASURED by a
    defect in the measuring apparatus? A broken judge, a raising guard, a crashed turn pass and
    a broken deterministic target check all score NOTHING, so none of them dents a pass rate —
    a run can be fully ``scored`` and still be describing a smaller surface than its eval set
    claims to cover. ``measurement_clean`` is THREE-valued for the same reason ``scored`` is
    derived from evidence: a run whose backend sent no block at all is UNKNOWN, not clean.

    ``ended_early`` names the rows whose dialog ended for something other than the user reaching
    their goal, with the typed reason per row. Such a scenario scores a low pass_rate for
    obligations its conversation never reached, and every consumer of this file — the report,
    the suite critic — reads that as the agent's failure unless the run's own verdict says
    otherwise. The reasons need DIFFERENT remedies, which is why the row carries the reason and
    not a flag: an exhausted cap is a budget the operator sets, an all-degraded dialog is a
    defect in ``simulate.py``, and a user who gave up is a finding about the SCENARIO.

    BEST-EFFORT by design: any write failure is logged at warning level and swallowed.
    This file is a report ABOUT the run — letting its write fail the run would invert the
    very classification it exists to provide (the same posture as the ``.run_mode`` /
    pin-recording / ``_resolve_dataset_url`` best-effort writes elsewhere in this tree).

    SIDE EFFECT, on the same best-effort terms: a run that RAN ROWS also republishes the
    workspace's ``.last_run`` pointer (:func:`_publish_run_pointer`) — an absolute path to this
    ``out``, written beside the ``--agent-module-path`` the caller named, never under ``out``
    itself. The rule is *iff this invocation ran rows*, read off ``verdict.ran``: a run with
    failing rows MOVES it (that is the run a repair flow wants) and a guard exit that
    republishes this summary having run nothing does NOT, so the pointer can never advance onto
    a refusal.
    """
    # A row the HARNESS failed was never scored: ``not_valid`` beside the verdicts that were. A row
    # that also returned a result (a redrive, a stability retry) keeps that result's verdict.
    scenario_verdicts = {**{sid: 'not_valid' for sid in _run_ledgers().harness_fault if sid in scenarios}, **(scenario_verdicts or {})}
    verdict = _resolve_run_verdict(
        out,
        scenarios=scenarios,
        graded=graded,
        cfg=cfg,
        ended_early=ended_early,
        ended_early_causes=ended_early_causes,
        measurement_integrity=measurement_integrity,
        user_sim_integrity=user_sim_integrity,
        eval_revisions=eval_revisions,
        degraded=degraded,
        echoes_dropped=echoes_dropped,
        turn_cap_reruns=turn_cap_reruns,
        timings=timings,
        stability=stability,
        family_gradient=family_gradient,
        world_unread=world_unread,
        tool_channel_unread=dict(_run_ledgers().tool_channel_unread),
        mock_unanswered=dict(_run_ledgers().mock_unanswered),
        mock_unanswered_unclaimed=dict(_run_ledgers().mock_unanswered_unclaimed),
        ipc_trimmed=dict(_run_ledgers().ipc_trimmed),
        scenario_verdicts=scenario_verdicts,
        scenario_high_severity=scenario_high_severity,
        plan_exhausted_disputed_turns=plan_exhausted_disputed_turns,
        agent_close_preempted_targets=agent_close_preempted_targets,
    )
    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(
            out / RUN_SUMMARY_FILENAME,
            {
                'total': verdict.total,
                # Three disjoint buckets summing to ``total``. ``incomplete`` is broken out
                # rather than folded into either: a time-budget row ran and was graded (so it is
                # not a failure) over a dialog the budget cut (so it is not a clean success), and a
                # quarantined row took no measurement at all (``_row_outcome``); a consumer that
                # cannot see the difference reports a truncated or refused suite as one or the other.
                'succeeded': verdict.succeeded,
                'failed': verdict.failures,
                'incomplete': verdict.incomplete,
                'exit_code': exit_code,
                # Why the run stopped: a mock abort is ``{reason, cause, scenario, tool, status_code, remedy}`` and scored nothing,
                # so a reader that finds it takes no pass rate from this file; a credit abort is
                # ``{reason: credit, code, resets_at?, scenario, remedy}`` and keeps the rows that finished. ``null`` otherwise.
                'aborted': dict(aborted) if aborted is not None else None,
                # The scenarios the abort kept from finishing: never started, cancelled, or the one that was refused.
                'not_run_scenarios': sorted(not_run),
                # What waiting for the run's own room cost; present only when a start waited.
                **({'credit_waits': waits} if (waits := _run_ledgers().credit_waits.to_record()) is not None else {}),
                'exit_code_covers': verdict.exit_code_covers,
                # WHICH invocation wrote this. The file is cumulative and is rewritten by every
                # attempt into the same --out, so without a stamp there is nothing separating a
                # verdict from the one it replaced — and an early guard exit would republish the
                # previous attempt's. ``run_name`` is the same handle the observability rows
                # carry (None when the run exited before one was resolved).
                'written_at': datetime.now(UTC).isoformat(timespec='seconds'),
                'run_name': run_name,
                # WHICH SDK ran this, and WHICH BACKEND scored it. Written always, ``null``
                # included for the backend, so "did two runs use the same binary against the
                # same backend?" is answerable from the artifacts rather than by inference.
                'sdk_identity': sdk_identity(),
                'backend_build': _aggregate_backend_build(),
                # WHICH rows the backend scored over a world it does not certify, and WHY — its
                # own typed map, because ``finalize.status`` says only whether a decisive policy
                # fired, and a reader that took ``'failed'`` for an untrustworthy world would
                # print "NEVER SCORED" over landed cells. Written always,
                # empty included, on the same rule as its neighbours.
                'untrustworthy_scenarios': dict(sorted(_run_ledgers().untrustworthy.items())),
                # HOST EFFECTS: which host-writing or external tools ran for real by the
                # operator's consent, and every one the harness denied and how. Written always; no
                # key but the disclosure ones means no harness reported, never that the agent was
                # confined. ``disclosure`` is read off the artifacts on disk, for every agent shape.
                # A partial re-run unions the prior block (an escape a superseded attempt saw stays escaped; a carried
                # row never reads confined by omission); the disclosure is read afresh over the merged rows, overlaid.
                'host_effects': {
                    **(_union_host_effects([verdict.prior_host_effects, _host_effects_summary()]) if verdict.prior_host_effects else _host_effects_summary()),
                    **_home_disclosure(out, verdict.scenarios, repo=_bench_repo(cfg)),
                },
                'scored': verdict.scored,
                'scored_this_invocation': verdict.scored_this_invocation,
                'eval_set_attached': verdict.eval_set_attached,
                # The evidence behind the two flags above, per row — so a partly-graded run is
                # readable as such instead of collapsing to a bare `scored: false`, and so a
                # later retry can carry these rows forward without re-asserting anything.
                'graded_scenarios': verdict.graded,
                # The rows the harness ended before either stop condition, and WHY. Written
                # always, empty included: "no scenario was cut" and "this writer did not
                # record cuts" are different claims, and only the first licenses reading a
                # low pass_rate as the agent's.
                'ended_early_scenarios': verdict.ended_early,
                # Per row whose simulated user STATED why it gave up, that cause — beside the
                # reason map rather than inside it, because ``{scenario: reason}`` is the shape
                # every consumer of that map reads. Written always, empty included, on the same
                # rule as its sibling: "no cause was stated" and "this writer records none" are
                # different claims. The reason says WHO stopped; the cause says what it saw.
                'ended_early_causes': verdict.ended_early_causes,
                # Per `plan_exhausted` row that ended on a simulated-user push-back, the 0-based agent turn it disputed.
                'plan_exhausted_disputed_turns': verdict.plan_exhausted_disputed_turns,
                # Per row the agent closed before its target beat arose, that beat.
                'agent_close_preempted_targets': verdict.agent_close_preempted_targets,
                # Per row that ended early, its ONE route `{owner, remedy, qualifier?}` — the run report renders these verbatim.
                'ended_early_routes': verdict.ended_early_routes,
                # The cut rows carried from a prior summary that holds no cause map: their
                # cause is unknown because no writer recorded one, which is not the backend
                # stating none. Written always, empty included, on the same rule.
                'ended_early_causes_unrecorded': verdict.ended_early_causes_unrecorded,
                # Which ids THIS invocation actually ran. With a merge the scenario map spans
                # more than one attempt, so without this there is nothing saying which rows are
                # fresh — and `exit_code` describes only the ids listed here.
                'ran_this_invocation': verdict.ran,
                # MEASUREMENT INTEGRITY. Written always, ``null`` included: "every row's
                # measurement was sound", "some row's was not" and "no row told us" are three
                # different claims, and a two-valued flag has to call the third one clean.
                'measurement_clean': verdict.measurement_clean,
                # WHICH channel decided it. Without this the flag reads as a contradiction of the
                # per-row map right below it — every row `clean: true` under a top-level `false` —
                # and the reader has no way to reach the user-sim block that actually drove it.
                'measurement_unclean_because': verdict.measurement_unclean_because,
                # WHICH channel could not be measured on some row — the reason a `null` above is a
                # coverage gap (an unread language on the evaluator-voice check) rather than an
                # older backend. Kept apart from the list above: a gap is read, a breach is fixed.
                'measurement_unchecked_because': verdict.measurement_unchecked_because,
                # Per row, the HOUSE verdict status (passed|failed|not_evaluated|unclean|unknown) the
                # headline tallies. `succeeded` above is row status: the thread completed.
                'scenario_verdicts': verdict.scenario_verdicts,
                # Per row whose verdict SERVED one, the backend's high-severity reds beside the goal; a
                # row an older backend scored is absent (unknown), never `[]`.
                'scenario_high_severity': verdict.scenario_high_severity,
                # WHICH eval set revision graded this run (the backend finalize's ``eval_revision``:
                # the content digest over the set's per-policy fingerprints). ``eval_set_id`` is
                # minted once and survives every in-place edit; this moves with the rubrics, so a
                # lane pin can tell "scored under the current judges" from "scored under the ones
                # an edit session since replaced". Null when no row carries one (an older backend)
                # or rows disagree; the per-row map beside it says which.
                'eval_revision': verdict.eval_revision,
                'eval_revisions': verdict.eval_revisions,
                # WHICH BYTES of the eval set this run was launched against:
                # `draft` (an edit session was open, so the rubrics were that session's
                # mid-edit working copy), `promoted`, `unknown` (the backend has no session
                # store), or `null` (a backend predating the field — never read as promoted).
                # Written always, empty included, on the same rule as `egress` and
                # `scan_contract_drift`: "the committed set", "a mid-edit draft" and "nobody
                # asked" are three different runs, and an absent key makes the third read like
                # the first. THIS IS THE ONLY KEY THAT ANSWERS IT for a live run: `eval_revision`
                # above comes from /finalize, which stamps a content digest with no draft
                # awareness (the `draft:<session_id>+<digest>` form is stamped by score_run's
                # replay path alone), so a run launched mid-session carries a digest of the draft
                # bytes that reads exactly like a committed set's.
                'eval_set_state': dict(_LAST_EVAL_SET_RECORD),
                # The DENOMINATOR behind the flag above. ``null`` collapses "no row carried a
                # block" and "2 of 3 rows did" into one token, and they are different runs: the
                # second measured most of itself and the first measured none of it. Written so a
                # reader can see the coverage without recomputing it against `scenarios`.
                'measurement_coverage': {'reported': len(verdict.measurement_integrity), 'population': verdict.total},
                'measurement_integrity': verdict.measurement_integrity,
                # SCENARIO FIDELITY — the user simulator's own per-turn anomalies, folded by the
                # backend. Its own key, not merged
                # into the block above: a broken judge and a simulator that skipped a scheduled
                # question have different owners and no shared remedy. It DOES ride
                # ``measurement_clean``, because a dialog driven off its scenario was not a
                # measurement of that scenario.
                'user_sim_integrity': verdict.user_sim_integrity,
                'user_sim_coverage': {'reported': len(verdict.user_sim_integrity), 'population': verdict.total},
                # The CLIENT-side integrity rows: per scenario, how many turns the HARNESS
                # emptied because the agent gave it nothing to send. Written always,
                # empty included, for the same reason ended_early is: "no turn was fabricated"
                # and "this writer did not record fabrications" are different claims, and only
                # the first licenses reading the transcript as the agent's own words.
                'degraded_scenarios': verdict.degraded,
                # The other client-side BREACH: per scenario, the tool calls that asked for ONE
                # seeded record and came back empty, and which tools made them. Written
                # always, empty included, on the same rule as its neighbour: "every read saw the
                # seeded world" and "this writer did not look" are different claims. Tool names
                # only — never the argument values, which are the dialog's.
                'world_unread_scenarios': verdict.world_unread,
                # Per scenario, the declared native tools whose calls no lane recorded: the row's
                # tool channel was not measured (``measurement_unchecked_because: tool_channel``).
                'tool_channel_unread_scenarios': verdict.tool_channel_unread,
                # Per scenario, the tool calls the mock never answered (the agent read a stub):
                # ``{calls, restart_shaped, served_schema_stub}``. Written always, empty included.
                # The typed per-call record is the run's ``mock.tool_call_unanswered`` finding.
                'mock_unanswered_scenarios': verdict.mock_unanswered,
                # Per scenario whose child claimed NOTHING about unanswered calls: ``true``. Not measured, never
                # clean (``measurement_unchecked_because: mock_unanswered``). Written always, empty included.
                'mock_unanswered_unclaimed_scenarios': verdict.mock_unanswered_unclaimed,
                # Per scenario, how many agent replies the child trimmed to fit one IPC line (their message chain never
                # reached the judges). Folds into ``measurement_clean``; the turns are the ``ipc.reply_trimmed`` finding.
                'ipc_trimmed_scenarios': verdict.ipc_trimmed,
                # What this host gave the agent of what its own files declare it needs, resolved
                # BEFORE the run (:func:`_preflight_host_toolchain`): each requirement's status and
                # the one host stamp a run with gaps carries. A measurement only; never an overrule.
                # Written always, empty included: "nobody asked" is not "nothing was missing".
                'host_toolchain': dict(_LAST_HOST_TOOLCHAIN_RECORD),
                # The OTHER client-side-only fact about the channel a judge reads: per scenario,
                # how many messages the single-role split took OUT of the agent channel, by rule
                # (``role_declared`` is the ordinary LangGraph shape; ``verbatim_echo`` is the
                # adapter returning the user's own words under no role). Written always, empty
                # included, for the same reason as its neighbours: a stderr log line per scenario
                # sits on a stream the routine `| tee run.log` drops.
                'echoes_dropped_scenarios': verdict.echoes_dropped,
                # The ceiling-hit re-runs this run performed (--rerun-on-turn-cap): both
                # budgets, the superseded thread, and whether the attempt it replaced had been
                # graded — so a row that reads worse after a re-run is attributable.
                'turn_cap_reruns': verdict.turn_cap_reruns,
                # ISOLATION. Written always, empty included, for the same reason
                # ``measurement_clean`` is: "isolation was established", "it was not" and "nobody
                # asked" are three different runs, and an absent key makes the third read like
                # the first. The full record — every declared host, every probe verdict — is in
                # egress_preflight.json beside this file.
                # A trial of a ``--trials N`` run reads the invocation's ONE guard log (``egress_dir``):
                # every agent child of every trial inherits the same guard and writes there.
                'egress': _egress_summary(egress_dir or out),
                # CORPUS DRIFT. Written always, empty included, on the same rule as
                # ``egress`` above: "the sources still match", "they drifted" and "nobody asked"
                # are three different runs, and an absent key makes the third read like the first.
                # The full comparison — every added/removed/changed file — is in
                # scan_contract_recheck.json beside this file.
                'scan_contract_drift': dict(_LAST_SCAN_DRIFT_RECORD),
                # WHAT THIS RUN COULD READ OF THE WORLD IT SCORES: the declared owner, whether the
                # harness hands its world over, and the warning text when neither can. Written
                # always, empty included, on the same rule as ``egress`` and its neighbours above
                # — the fourth axis of exactly that shape. Without it a report sees N world-reading policies at not-measured with
                # no reason on file and blames the backend or the eval set (the demotion itself
                # happens in ``child.handle_finalize`` + the backend's ``world_unobservable``).
                'world_observability': dict(_LAST_WORLD_RECORD),
                # An explicit opt-in to live writes on a world nothing resets per
                # conversation RUNS, carrying the `shared_world` named risk (the suite caps at
                # qualified) and `stability_comparable: false` (:func:`_shared_world_summary`).
                **_shared_world_summary(),
                # SOLVABILITY: which of this run's scenarios the suite had already
                # decided it cannot honestly measure. Written always, ``read: false`` included —
                # "none is quarantined" and "the census was never taken" are different runs, and
                # a report that cannot tell them apart attributes a refusal to the agent.
                'solvability': dict(_LAST_SOLVABILITY_RECORD),
                # The SET's shape (requested vs shipped, demands delivered short). Written always; ``{}`` = not read.
                'scenario_set': dict(_LAST_SCENARIO_SET_RECORD),
                # THE TURN CAP this run played at, and where it came from (``flag`` / ``derived`` from the
                # run's scenario floors / ``fallback`` when they could not be read). ``null`` on an exit
                # before the cap was resolved.
                'max_turns': cfg.max_turns,
                'max_turns_source': cfg.max_turns_source,
                # The STABILITY retry's record, per row that earned one. Present
                # only for rows that were in doubt: a missing row means the scenario was never
                # in doubt, NEVER that its failure was proven stable. Each row carries the
                # scenario verdict (``stability``) AND its per-policy grain
                # (``policy_reproduction``: per red, reproduced / not_reproduced / retry_only /
                # unmeasured) — the verdict is derived from the map, so a reader acting on ONE
                # red can see whether that red was the one that reproduced. A refused retry
                # carries the typed ``shed_reason`` (and ``shed_signals`` for the shed decided
                # on infrastructure counts) beside its sentence. Each row also states the eval
                # revision that graded each attempt (``first_eval_revision`` /
                # ``second_eval_revision``) and, ONLY as a disproof, ``marks_valid: false`` with
                # its reason — the marks compare two attempts, so they mean something only under
                # one revision (:func:`_marks_comparability`).
                #
                # How a FLAKY failure is PRESENTED to the customer (`fail (flaky)`, `1/2`, or suppressed
                # with a footnote) is not decided here. Retrying only failures is statistically biased
                # (passes are never re-tested), so anything that lets the second attempt rescue the first
                # inflates the suite score: the first attempt stays authoritative everywhere.
                'stability': verdict.stability,
                # Per-family pass rates. A `0/7` says only "everything failed"; 0% in
                # one family and 100% elsewhere is a defect with an address.
                'family_gradient': verdict.family_gradient,
                # Per row: when it started, when it ended, and how long it took — the WHOLE row,
                # re-drives and ceiling re-runs included. Written always, empty included, so
                # "which scenario was the slow one" needs no comparison of artifact file mtimes.
                'scenario_timings': verdict.timings,
                'scenarios': verdict.scenarios,
                # How this run replayed its scenarios — read by :mod:`plurai.trials` for an attempt whose
                # agent raised, which leaves no verdict to read the mode off.
                'replay_mode': _replay_mode_of_run(cfg),
                # The scenarios the replay plan played single-hop beside the others (``--replay-plan``); written always.
                'single_hop_scenarios': (hops := _single_hop_scenarios(cfg, verdict.scenarios)),
                # ``totals`` and the REPLAY RATES over this run's own cells: ``policy_turn_rates`` always,
                # and ``regression_replays`` when a recorded regression ran (:func:`_cell_blocks`).
                # A ``--trials N`` run recomposes all three over every trial in :func:`_publish_trials`.
                **(
                    cell_blocks := _own_cell_blocks(
                        out,
                        {
                            'scenarios': verdict.scenarios,
                            'scenario_verdicts': verdict.scenario_verdicts,
                            'replay_mode': _replay_mode_of_run(cfg),
                            'single_hop_scenarios': hops,
                        },
                    )
                ),
                # The likelihoods the suite score weighs by, and THE ``scores`` block over this run's verdicts
                # (:mod:`plurai.scoring`). A ``--trials N`` run recomposes it over every trial in :func:`_publish_trials`.
                **_own_scores(cfg, out, verdict.scenarios, cell_blocks),
                # ``$OUT``'s ``trials`` block names the trial directories every reader enumerates;
                # any other write of this file in a ``--trials N`` run (an early exit, trial 1's
                # own summary) keeps it until :func:`_publish_trials` recomposes it.
                **_carried_trials_block(cfg, out),
            },
        )
        if cfg.trial is None:
            # A ``--trials N`` run says it once, over every trial, in :func:`_publish_trials`.
            _warn_reduced_restores(cell_blocks)
    except Exception:  # noqa: BLE001 — a report about the run must never BE the run's failure
        logger.warning('could not write %s to %s', RUN_SUMMARY_FILENAME, out, exc_info=True)
    # The workspace's pointer AT this run, written beside the verdict and gated on the one
    # already-typed fact that means "rows ran here". ``verdict.ran`` and NOT
    # ``exit_code``: exit 1 is both "ran, some rows failed" — precisely the run the repair
    # flow exists to read — and "refused before anything ran", while ``ran`` is ``[]`` for
    # exactly the guard exits, because :func:`_publish_early_exit` passes ``scenarios={}``.
    #
    # OUTSIDE the summary's guard, and that placement is the whole of it: inside, every failure
    # this call can raise that its own catch does not name would be reported as "could not write
    # run_summary.json" — a file that has already been written successfully one line earlier.
    # The two failures have different remedies and different blast radii (a stale ``.last_run``
    # makes the next ingest-run / report / aggregate describe a DIFFERENT run), which is the
    # reason :func:`_publish_run_pointer` carries its own warning at all. Best-effort on the
    # same terms, so a run that finished still finishes.
    # A TRIAL of a ``--trials N`` run never publishes it: N concurrent trials would race for one
    # pointer, and the one it must name is ``$OUT`` — published once, by :func:`_publish_trials`.
    # ``--no-run-pointer`` (a side run into its own --out, e.g. regression replays) leaves it alone.
    # A run stopped for credit, capacity or an unavailable backend is not the run the next command should describe, so it leaves the pointer as it was.
    measured_stop = isinstance(aborted, Mapping) and aborted.get('reason') in ('credit', 'capacity', 'unavailable')
    if verdict.ran and cfg.trial is None and not cfg.no_run_pointer and not measured_stop:
        _publish_run_pointer(cfg, out)
    return verdict


def _replay_mode_of_run(cfg: _RunConfig) -> str:
    """``single_hop`` for a ``--replay-turn-only`` run, else ``full`` (:data:`plurai.trials.ReplayMode`); a run whose
    plan played some scenarios single-hop names them in ``single_hop_scenarios`` (:func:`_single_hop_scenarios`)."""
    return 'single_hop' if cfg.replay_turn_only else 'full'


def _single_hop_scenarios(cfg: _RunConfig, scenarios: Mapping[str, Any]) -> list[str]:
    """The scenarios of this run the replay plan played single-hop, sorted; ``[]`` for a run-level mode."""
    return [] if cfg.replay_turn_only else sorted(sid for sid in scenarios if sid in cfg.single_hop_ids)


def _warn_reduced_restores(blocks: Mapping[str, Any]) -> None:
    """One warning line per recorded regression a replay of which resumed with LESS state than the recording had
    (``regression_replays`` → ``restore_reduced``, counted in every mode): what it did at the cut is no reading of the
    recorded defect. A single hop's cut reads ``inconclusive`` there; a whole replay keeps its later turns."""
    block = blocks.get('regression_replays')
    rows = block.get('regressions') if isinstance(block, Mapping) else None
    for sid, entry in sorted((rows or {}).items()):
        reduced = entry.get('restore_reduced') if isinstance(entry, Mapping) else None
        if isinstance(reduced, int) and reduced > 0:
            print(
                f'[regressions] WARNING: {sid}: {reduced} of {entry.get("valid")} replay(s) resumed with less state than the '
                "recording had (the harness's seed_state dropped resumed state) — not measured at the cut.",
                file=sys.stderr,
            )


def _cell_blocks(cells_by_trial: Sequence[Mapping[str, ScenarioCells]]) -> dict[str, Any]:
    """The summary's keys computed over ``cells_by_trial`` (:mod:`plurai.trials`): ``totals`` (:func:`run_totals`)
    and ``policy_turn_rates`` always, and ``regression_replays`` only when a recorded regression ran."""
    blocks: dict[str, Any] = {'totals': run_totals(cells_by_trial), 'policy_turn_rates': aggregate_policy_turn_rates(cells_by_trial)}
    replays = aggregate_regression_replays(cells_by_trial)
    if replays['regressions']:
        blocks['regression_replays'] = replays
    return blocks


def _own_cell_blocks(out: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`_cell_blocks` over ONE run: the cells of the scenarios ``summary`` names, read
    from ``out`` (:func:`plurai.trials.cells_from_summary`). Best-effort on the summary's own terms:
    a failure here leaves the keys out, never the summary unwritten."""
    try:
        return _cell_blocks([cells_from_summary(out, dict(summary))])
    except Exception:  # noqa: BLE001 — a rate about the run must never cost the run its summary
        logger.warning('could not compute the totals and replay rates for %s', out, exc_info=True)
        return {}


def _scores_or_nothing(out: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """``{'scores': ...}`` (:func:`plurai.scoring.run_scores`) over the population ``summary`` names; ``{}`` when it
    cannot be computed — a score about the run must never cost the run its summary."""
    try:
        return {'scores': scoring.run_scores(out, summary)}
    except Exception:  # noqa: BLE001 — reported; the summary is written without the block
        logger.warning('could not compute the scores for %s', out, exc_info=True)
        return {}


def _own_scores(cfg: _RunConfig, out: Path, scenarios: Mapping[str, Any], cell_blocks: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded likelihoods and the ``scores`` block for ONE run's summary. A trial ``k > 1`` of a ``--trials N``
    run writes neither: ``$OUT``'s composed summary holds the one block, over every trial. ``cell_blocks``' run totals
    take their pass count off the block (:func:`plurai.scoring.align_totals`, in place)."""
    if cfg.trial is not None and cfg.trial.index != 1:
        return {}
    likelihoods = dict(_LAST_SCENARIO_LIKELIHOODS)
    scored = _scores_or_nothing(out, {'scenarios': scenarios, scoring.LIKELIHOODS_KEY: likelihoods})
    scoring.align_totals({'totals': cell_blocks.get('totals'), **scored})
    return {scoring.LIKELIHOODS_KEY: likelihoods, **scored}


def _carried_trials_block(cfg: _RunConfig, out: Path) -> dict[str, Any]:
    """``{'trials': <the block already in out's summary>}`` when this write replaces ``$OUT``'s
    summary in a ``--trials N`` run, else ``{}``."""
    if cfg.trials < 2 or (cfg.trial is not None and cfg.trial.index != 1):
        return {}
    prior = _read_json_object(out / RUN_SUMMARY_FILENAME)
    block = prior.get('trials') if prior is not None else None
    return {'trials': block} if isinstance(block, dict) and block.get('schema') == TRIALS_SCHEMA else {}


def _clear_stale_run_summary(out: Path, *, partial: bool = False) -> None:
    """Drop a previous run's ``run_summary.json`` at strategy entry — and the eval-set record with it.

    Without this, an early guard exit (unresolvable scope, a rejected eval set) leaves
    LAST run's verdict sitting in ``--out`` for a reader to mistake for this one's — the
    same stale-artifact trap ``_run_one`` already guards per scenario. Best-effort: an
    unlink failure is not a reason to refuse a run.

    PRECONDITION: :func:`_occupied_out_refusal` has already run in ``main()``. The only
    summaries this delete can still meet are a partial run's cumulative file (kept below), and
    one the operator authorised with ``--replace`` — never a finished run nobody asked to
    discard. A caller that reaches a strategy WITHOUT going through ``main()`` has skipped that
    rung and is deleting a measurement.

    ``partial`` (a ``--scenario-id`` run) KEEPS the file: a targeted re-run is not a new
    population, it is a correction to an existing one, and the scenario directories of the
    ids it is not running are still sitting in ``out`` for the aggregator to count. Deleting
    the prior verdict there is what leaves the retry unable to merge — see
    :func:`_merge_prior_scenarios`.

    Keeping the MAP is right; keeping the VERDICT is not, and this function cannot fix that
    — at strategy entry there is no exit code yet to replace it with. An early exit on a
    partial run is therefore closed at the exit itself, by :func:`_publish_early_exit`.

    THE EVAL-SET RECORD is cleared here UNCONDITIONALLY (partial runs included), because it is a
    module-level handoff rather than a file: the pre-flight clears it too, but it runs AFTER the
    early exits that can publish a summary, so a second invocation in one process would republish
    the previous run's eval-set state beside this run's verdict.
    """
    _LAST_EVAL_SET_RECORD.clear()
    if partial:
        return
    try:
        (out / RUN_SUMMARY_FILENAME).unlink(missing_ok=True)
    except OSError:
        logger.debug('could not remove a stale %s in %s', RUN_SUMMARY_FILENAME, out, exc_info=True)


def _prior_run_detail(path: Path) -> str:
    """WHOSE run the summary in ``--out`` says it is — for the MESSAGE only, never the decision.

    Best-effort on purpose, and the one place this module reads a prior summary without caring
    whether it can: the refusal below has already been decided by the time this runs, so an
    unreadable file degrades the sentence rather than the outcome.
    """
    try:
        prior = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        prior = None
    if not isinstance(prior, dict):
        return 'its run name and timestamp could not be read'
    return f'run_name={prior.get("run_name")!r}, written_at={prior.get("written_at")!r}, {prior.get("total")} scenario(s)'


def _occupied_out_refusal(cfg: _RunConfig) -> str | None:
    """REFUSE a whole-set run into an ``--out`` that already holds a run's verdict — the THIRD
    disposition, beside the merge (:func:`_merge_prior_scenarios`) and the clear
    (:func:`_clear_stale_run_summary`).

    PRESENCE is the predicate, not a grading of the file's contents. The producer writes
    ``run_summary.json`` at exactly one moment — when a run finalises, or when a partial run's
    early exit re-stamps its own — so the file BEING there is already the fact "a run was
    measured into this directory", stated by the only component entitled to state it. Asking
    instead whether it "describes a finished run" invents a second closed notion that will
    drift from the writer, and it lets through the one prior that cannot defend itself: a torn
    or half-written file, where an operator most needs to be stopped. Hence ``is_file()`` and
    nothing else; :func:`_prior_run_detail` parses only to say whose run is in the way.

    The two invocations that legitimately re-enter an occupied directory are read off the
    TYPED config rather than re-derived: ``is_partial`` (a ``--scenario-id`` run is a
    correction to the population already there, and merges into it) and ``replace`` (the
    operator said to discard it). This rung is the clear's precondition.

    The per-scenario sibling already makes this argument one level down:
    :func:`_displace_prior_attempt` moves a scenario's previous artifacts into
    ``previous_attempt/`` rather than deleting them.
    """
    if cfg.is_partial or cfg.replace:
        return None
    prior = cfg.out / RUN_SUMMARY_FILENAME
    if not prior.is_file():
        return None
    return (
        f'{cfg.out} already holds a finished run ({_prior_run_detail(prior)}), and running the whole set here would '
        f'DELETE its {RUN_SUMMARY_FILENAME} — the only record of a run that was already paid for. Either pass a fresh '
        f'--out, or name the ids with --scenario-id (a targeted run MERGES into the existing summary instead of '
        f'replacing it), or pass --replace to discard that run on purpose.'
    )


def _invariant_exit(cfg: _RunConfig | None) -> int | None:
    """After a rung that later rungs depend on: the funnel's refusal code when a HARD finding was reported,
    else ``None`` (advisory findings accumulate until the funnel). ``cfg`` is ``None`` before the run owns
    ``--out``, so nothing is re-published there."""
    if not any(f.authority in HARD_AUTHORITIES for f in current_gate_authority().findings):
        return None
    if cfg is None:
        return held_exit(refusal_code=exit_codes.REFUSED)
    return _held_by_gates(cfg)


def _relative_to_bench(path: Path, cfg: _RunConfig) -> str:
    """``path`` relative to the bench root, for stable finding evidence (an absolute path moves between machines)."""
    try:
        return str(Path(path).resolve().relative_to(_bench_repo(cfg).resolve()))
    except ValueError:
        return Path(path).name


def _unreadable_declaration(declaration: str, exc: Exception) -> dict[str, Any]:
    """The finding fields for a declaration every downstream reader would raise on; the caller names the gate."""
    return {
        'subject': declaration,
        'observed': {'error': str(exc)},
        'remedy': f'{declaration} could not be read, so this run cannot establish its shape or whether its world is resettable: {exc} '
        'Fix or remove the config.',
        'draft_ref': None,
    }


def _held_by_gates(cfg: _RunConfig) -> int | None:
    """THE gate funnel at a pre-spend point: the refusal code when a finding reported so far holds the run
    (a security or invariant one, or a holding advisory one no ``--overrule`` lifted), after re-publishing
    the summary for an exit that ran nothing; ``None`` otherwise. A finding that holds nothing never stops a run."""
    if (held_code := held_exit(refusal_code=exit_codes.REFUSED)) is None:
        return None
    _publish_early_exit(cfg, held_code)
    return held_code


def _publish_early_exit(cfg: _RunConfig, exit_code: int) -> None:
    """Re-publish the summary for a guard exit that ran NOTHING.

    A full run's file was already dropped at strategy entry, so this is a no-op there. A
    ``--scenario-id`` run keeps its file — the map describes scenario directories still on
    disk and the merge depends on it — but the VERDICT in it belongs to the invocation that
    wrote it. Leaving that untouched republishes ``exit_code: 0, failed: 0, scored: true``
    for an attempt that refused to start (a rejected eval set, a failed dataset upload), and
    the payload carries no timestamp or run name for a reader to tell the two apart.

    Rewriting keeps the map and replaces the verdict with this invocation's: an empty
    ``ran_this_invocation`` says outright that no row here is fresh, ``exit_code`` is the
    code this call is about to return, and ``written_at`` re-stamps the file.

    Only ever OVERWRITES. With no prior file there is no stale verdict to correct, and a
    refusal that ran nothing should leave the same nothing behind that a full run's
    clear-at-entry does — not a fresh ``total: 0`` summary nobody asked for.
    """
    if not cfg.is_partial or not (cfg.out / RUN_SUMMARY_FILENAME).exists():
        return
    _write_run_summary(cfg.out, exit_code=exit_code, scenarios={}, graded=set(), cfg=cfg)


#: Sentinel distinguishing "the server did not send this key" from "the server sent []".
_ABSENT = object()


def _policy_name_list(per: dict[str, Any], key: str) -> list[Any] | None:
    """One of ``policy_evaluation_results``' policy-name lists, or ``None`` when the
    server did not send it.

    ``[]`` and ``None`` mean DIFFERENT things and the artifact must keep them apart:
    ``[]`` is the server stating "zero policies in this bucket" — a real measurement a
    reader can reconcile against ``pass_rate``. ``None`` is "this server never told
    us", which is what an older backend (or a trust-fail / no-policy finalize path
    that skips the key) produces. Collapsing them — the ``.get(key) or []`` idiom —
    manufactures the measurement: a reader sees ``errored_policies: []`` and concludes
    no eval was broken, when in fact nothing was ever reported. The backend takes the same
    posture on its side of the wire for ``skipped_policies``, surfacing the skip set even
    when empty so a reader can tell "0 skipped" from "field absent"; this is the client half
    of that contract.

    An explicit ``null`` on the wire is treated as absent — it carries no measurement
    either.
    """
    value = per.get(key, _ABSENT)
    if value is _ABSENT or value is None:
        return None
    return list(value)


def _run_context(
    cfg: _RunConfig,
    settings: Settings,
    *,
    dataset_name: str | None,
    run_name: str | None,
    started_at: str,
    max_turns: int | None = None,
    world_slot: int | None = None,
) -> dict[str, Any]:
    """The invocation's identity block, folded into transcript metadata (self-documenting runs).

    Everything here is already resolved in-process; writing it down is what makes a run
    scored against the wrong model or backend detectable afterwards. The agent's
    own model/provider stays out of reach by design (simulate.py is an opaque child that
    inherits the environment); the closest recorded signal is per-turn ``agent_state`` when
    the customer's agent returns one, plus this block's build tag.

    ``run_name`` is the RESOLVED per-invocation handle (``_unique_run_name`` — the value the
    experiment/dataset rows carry; a trial's own ``<run>-t<k>`` under ``--trials N``), threaded from the strategy rather than re-derived: this
    key is what joins an artifact to its LangSmith/Langfuse row, so recording the dataset
    name under it would leave the artifact unjoinable. ``started_at`` is captured at SCENARIO start in ``_run_one`` — a stamp taken at
    artifact-write time records a long scenario's FINISH as its start.
    """
    return {
        'agent_version': cfg.agent_version,
        # WHICH SDK wrote this file, from the LOADED module. Not ``__version__``:
        # that is dist metadata, which a stale editable install leaves out of date.
        'sdk_identity': sdk_identity(),
        'scenario_set_id': cfg.scenario_set_id,
        'eval_set_id': cfg.eval_set_id,
        'run_name': run_name,
        'dataset_name': dataset_name,
        'mock_service_url': cfg.mock_service_url,
        # THIS ATTEMPT's budget, not the CLI flag's. A ceiling re-run runs at a raised one, and
        # stamping ``cfg.max_turns`` there would put two contradicting numbers in one file:
        # ``max_turns: 12`` beside ``turn_cap_rerun: {max_turns: 24}``.
        'max_turns': cfg.max_turns if max_turns is None else max_turns,
        'server_url': settings.endpoint.url,
        'http_timeout': settings.network.http_timeout,
        'started_at': started_at,
        # WHICH TRIAL of a ``--trials N`` invocation wrote this file; absent on a single run.
        **_trial_facts(cfg),
        # WHICH WORLD COPY this attempt ran on (``world_state.slots``); absent with one shared world.
        **({'world_slot': world_slot} if world_slot is not None else {}),
    }


def _trial_facts(cfg: _RunConfig) -> dict[str, Any]:
    """``trial_index`` / ``trial_count`` / ``trial_group`` for a trial body's config, else ``{}``.

    One spelling for the transcript's ``metadata`` and the sinks' run facts
    (``eval_mapping.RUN_FACT_KEYS``), so an artifact and its observability record name the same trial.
    """
    if cfg.trial is None:
        return {}
    return {'trial_index': cfg.trial.index, 'trial_count': cfg.trial.count, 'trial_group': cfg.trial.group}


def _scenario_run_facts(cfg: _RunConfig, *, run_name: str | None, scenario_id: str) -> dict[str, Any]:
    """The run facts a scenario's sink record carries (``eval_mapping.RUN_FACT_KEYS``).

    ``run_id`` is ``run_name`` (the per-invocation handle written to ``run_summary.json`` and the
    transcript metadata), which is what joins a trace in the vendor UI to the local artifacts. It is
    not the ``experiment`` key ``build_scenario_metadata`` stamps: that one is the sink's own
    experiment/project label, which LangSmith can derive apart from the run name. This runner
    addresses a scenario by its folder name, so the scenario id IS the folder; both keys are
    stamped because readers search a trace store by either spelling.
    """
    return {
        **_trial_facts(cfg),
        **({'run_id': run_name} if run_name else {}),
        'scenario_id': scenario_id,
        'scenario_folder': scenario_id,
        'scenario_set_id': cfg.scenario_set_id,
        'sdk_version': sdk_identity()['version'],
    }


def _write_success_artifacts(
    scenario_out: Path,
    *,
    scenario: str,
    result: ThreadResult,
    run_context: dict[str, Any] | None = None,
    redrive: RedriveProvenance | None = None,
    attempt: _AttemptProvenance | None = None,
    turn_cap_rerun_failed: dict[str, Any] | None = None,
) -> None:
    """Persist the success artifacts in the schema the diagnose flow consumes.

    ``run_context`` is the invocation's identity block folded into ``metadata`` — see
    ``_run_context``: without it a run scored against the wrong model/backend/set is
    undetectable from the artifacts alone (a repo .env silently swaps
    the agent's model and nothing on disk shows it)."""
    scenario_out.mkdir(parents=True, exist_ok=True)
    per = result.finalize.policy_evaluation_results or {}
    per_turn_step_info = getattr(result.finalize, 'per_turn_user_step_info', None)
    _write_json(
        scenario_out / 'transcript.json',
        {
            # FIRST key, so a reader that peeks at the head of a large file finds it. The shape
            # below has six overlapping views of one conversation and no self-description; an
            # UNSTAMPED file is schema 1.
            # plurai.run_artifacts.load_transcript is the reader that owns the projection.
            'transcript_schema': TRANSCRIPT_SCHEMA,
            # ``type`` is the LANGCHAIN discriminator ('human' / 'ai' / 'tool'), matching what the
            # rich chain below carries — see LC_MESSAGE_TYPES. Spelling it as CLASS NAMES in
            # these two arrays describes one conversation in two vocabularies within one file,
            # depending on whether the agent happened to return a message chain.
            'user_messages': [{'type': LC_MESSAGE_TYPES['user'], 'content': m} for m in result.user_messages],
            # Rich agent chain (tool-call ai messages + tool messages) when available,
            # else the plain ai_text turns.
            'chatbot_messages': result.chatbot_full_messages or [{'type': LC_MESSAGE_TYPES['agent'], 'content': m} for m in result.chatbot_messages],
            # Turn-ordered, role-tagged log.
            # Each entry: turn_idx, user_message, agent_message, turn_messages (the turn's new chain
            # incl. tool calls, agent side only — see orchestrator._fresh_turn_tail and
            # split_user_echo), output_user_message, should_terminate. Free of the flat arrays'
            # positional off-by-one, and attributes tool calls to the turn that made them.
            'turns': result.transcript_turns,
            # turn_count == len(turns) by definition — the closing-turn reply lives in
            # `turns` but is NOT a /user-turn cycle, so `result.turns` can be one lower;
            # the server-authoritative cycle count keeps its own key.
            'turn_count': len(result.transcript_turns),
            'user_turn_cycles': result.turns,
            # Per-turn user-simulator step snapshots from the finalize response (str turn_idx
            # keys) — what makes turn_based policies RE-scoreable from this artifact via
            # the backend score_run. Absent key = older backend that never sent it.
            **({'per_turn_user_step_info': per_turn_step_info} if per_turn_step_info else {}),
            # What was INJECTED into this scenario at its start, in ONE block so a reader asking
            # "what did this run actually put in front of the agent?" has one place to look.
            # ``initial_agent_context`` is what the backend sent: a served document is its
            # ``<scheme>:///documents/…`` reference, never the bytes the agent receives, so a re-post of this
            # file (a judge replay, a rescore) stays under the request limit. Written ALWAYS, empty dicts included: a scenario seeded
            # with nothing and a writer that recorded nothing are different facts, and only the
            # first licenses reading a transcript as the whole input. The only other record is the
            # observability sink's ``example_inputs``, which a run with no sink never writes, so
            # without these a finished run could not be audited for its own inputs and an offline
            # re-score would read a conversation without half of what produced it.
            #
            # ``resume_seed_present`` is a FLAG, not the seed: the seed itself can be a whole
            # conversation, it already sits VERBATIM under the top-level ``resume_seed`` key below
            # (and in the scenario folder's resume_seed.yaml), and duplicating it here would make
            # one file carry the same conversation twice. The flag is the one bit this block owes
            # a reader — "was this a resumed leg?" — answerable without loading the seed.
            'injected_inputs': {
                'initial_world_state': result.initial_world_state,
                'initial_agent_context': result.initial_agent_context,
                # The keys of it the harness never read (``None`` = not recorded) — the re-score's input.
                'unread_inputs': result.unread_inputs,
                # The world right after ``seed_state`` (``None`` = not read) — the re-score's start world.
                'post_seed_world_state': result.post_seed_world_state,
                'resume_seed_present': result.resume_seed is not None,
            },
            # WHICH ROUTE drove this scenario, and — for a ONE-SHOT (pipeline) scenario — the hop
            # itself. Both keys are ABSENT on a dialog run, so a dialog transcript carries nothing
            # extra.
            #
            # The transcript is the CARRIER: ``score_saved_run`` receives this file and nothing
            # else, so a hop whose IO is not in it can never be re-scored, fixture-gated or
            # replayed. A hop with no transcript.json would be invisible to an ecosystem where a
            # missing transcript means "errored / incomplete".
            #
            # NO ``inputs`` key inside ``pipeline_io``: the hop's inputs ARE
            # ``injected_inputs.initial_agent_context`` directly above — /finalize sends that dict
            # as ``pipeline_inputs`` — and one truth gets one carrier in one file. A re-score binds
            # ``pipeline_inputs`` from there and ``pipeline_outputs`` / ``pipeline_error`` from
            # here. (The standalone ``pipeline_io.json`` twin below restates the inputs because it
            # is a file opened on its own, not because the value lives twice in this one.)
            **({'scenario_shape': result.scenario_shape} if result.pipeline_io is not None else {}),
            **({'pipeline_io': result.pipeline_io} if result.pipeline_io is not None else {}),
            # The resume seed this scenario ran from, VERBATIM, present ONLY for a
            # trace-resumed scenario. TOP-LEVEL, a peer of the message arrays, because it
            # answers the same question they do — what conversation was this? — while
            # ``metadata`` is the identity block, not conversation content. Additive and
            # optional, so no TRANSCRIPT_SCHEMA bump (see run_artifacts.TRANSCRIPT_SCHEMA).
            #
            # VERBATIM is the load-bearing word, not tidiness: the live run built its turn-0
            # excision baseline from these MINIMAL messages, and ``convert_to_messages`` is
            # NOT idempotent on its own full dump — re-feeding a ``model_dump`` folds
            # ``invalid_tool_calls`` / ``usage_metadata`` into ``additional_kwargs``. A
            # baseline rebuilt from that would be a DIFFERENT list, would stop prefix-matching
            # what the run recorded, and an offline re-score would read the seeded history as
            # the agent's own turn 1 — a green re-score of a different conversation. So this
            # is exactly what the backend sent, never a dump of what it was converted into.
            **({'resume_seed': result.resume_seed} if result.resume_seed else {}),
            # ``seed_state``'s own account of what it restored from that seed, present ONLY
            # when the harness reported a REDUCED restore — a faithful one reports nothing.
            # A peer of ``resume_seed`` because it QUALIFIES it: the seed is the conversation
            # the scenario was supposed to start from, this is how much of it the agent
            # actually started with, and a messages-only restore that reads as a faithful one
            # is scored as the conversation the seed describes rather than the one that ran.
            # The backend echoes the same report back on the finalize response, but NOTHING on
            # disk carries that echo — the artifacts persist the finalize payload's world
            # state, tool events and eval block, never its resume fields — so this key is the
            # run's only local record of restore fidelity. Additive and optional, so no
            # TRANSCRIPT_SCHEMA bump (see run_artifacts.TRANSCRIPT_SCHEMA).
            **({'resume_fidelity': result.resume_fidelity} if result.resume_fidelity else {}),
            # ``redrive`` is present ONLY when this scenario was re-driven after a
            # restart-classified failure. It names the thread that died
            # and what classified it, so an absorbed false-positive stays readable
            # in the artifact a human opens — not only in a log that may be gone.
            'metadata': {
                'scenario': scenario,
                'thread_id': result.thread_id,
                # The eval set revision that graded THIS row (see ``_eval_revisions``); absent on
                # an older backend, never defaulted.
                **({'eval_revision': revision} if (revision := _eval_revision_of(result)) else {}),
                # Present ONLY when the HARNESS ended the dialog (the turn cap running out
                # mid-conversation, or every turn emptied by the parent) rather than the user
                # simulator's own should_terminate. A transcript that stops otherwise reads as a scenario that
                # resolved, which is the opposite verdict.
                **({'ended_early': result.ended_early} if result.ended_early else {}),
                # The simulated user's own stated CAUSE for the reason above, when it gave up
                # and the backend reported one. Present ONLY beside a reason: it is what tells a
                # reader "the agent refused" from "the user never saw the reply", and without it
                # every give-up reads as the former.
                **({'ended_early_cause': result.ended_early_cause} if result.ended_early and result.ended_early_cause else {}),
                # The server-side finalize job this scenario submitted. One per scenario by
                # construction (a finalize is thread-scoped and each scenario owns one thread) —
                # written down so that property is CHECKABLE from a finished run rather than
                # assumed.
                **({'finalize_job_id': result.finalize_job_id} if result.finalize_job_id else {}),
                **({'redrive': redrive.to_artifact()} if redrive else {}),
                # The turns the HARNESS emptied, ``{turn_idx, subkind}`` each. Present ONLY when
                # some reply in this very file was manufactured by the runner rather than given
                # by the agent — a judge's verdict on those turns may not be about the agent (the
                # per-turn warning names the owners to check), and nothing else in the artifact
                # distinguishes them from an agent that chose silence. When EVERY turn is listed
                # here, ``ended_early`` says ``adapter_degraded`` beside it and the file is not a
                # measurement of the agent.
                **({'degraded_turns': result.degraded_turns} if result.degraded_turns else {}),
                # The turns whose reply the child TRIMMED to fit one IPC line: the reply text is the agent's, its message
                # chain is not in this file. Present ONLY when some turn was trimmed.
                **({'ipc_trimmed_turns': result.ipc_trimmed_turns} if result.ipc_trimmed_turns else {}),
                # Present ONLY when the dialog LANDED on a failed turn: ``{turn_idx, closing, kind,
                # exception, child_exc_class, error}``. Mid-dialog, ``ended_early`` names the failure's
                # owner and the judges graded the turns before it; on the closing call the dialog had
                # ended, and they graded it one reply short.
                **({'turn_failure': result.turn_failure} if result.turn_failure else {}),
                # Present ONLY when a finalize snapshot channel could not be read: it shipped unread, so
                # the policies that read it abstained (``{'world'?: detail, 'events'?: detail}``).
                **({'capture_failed': result.capture_failed} if result.capture_failed else {}),
                # Present ONLY when the harness's world snapshot carried values in their JSON form
                # (``{path: source type}``): a datetime as ISO text, a Decimal as a number, ….
                **({'world_values_coerced': result.world_values_coerced} if result.world_values_coerced else {}),
                # Present ONLY when the agent changed a file routing its vendor calls to the mock: calls after
                # the change may have reached the real service, and the world holds none of their effects.
                **({'interception_changed': result.interception_changed} if result.interception_changed else {}),
                # What the single-role split took OUT of this file's agent channel, by the rule
                # that moved it (``role_declared`` — the ordinary LangGraph shape; or
                # ``verbatim_echo`` — a role-less message that is verbatim what the harness sent).
                # Present ONLY when something was dropped. Here rather than only in a log, because
                # the question it answers is asked of the ARTIFACT: this file's chatbot_messages
                # is not the whole chain the adapter returned, and nothing else in it says so.
                **({'user_echoes_dropped': result.user_echoes_dropped} if result.user_echoes_dropped else {}),
                # Where this attempt sits in the folder's history — see _AttemptProvenance, the
                # same block error.json carries.
                # ``without_stale``: this call is about to write every name in
                # SCENARIO_ARTIFACT_FILES, so any of them the pre-attempt scan called stale is
                # stale no longer — only a file this writer does NOT touch (error.json) can
                # still belong to an earlier attempt.
                **(attempt.without_stale(SCENARIO_ARTIFACT_FILES).to_metadata() if attempt else {}),
                # This row IS the earlier attempt: the re-run at the raised budget failed and
                # the measurement it superseded was restored. Without it, a row that reads
                # ``turn_cap_rerun`` beside a transcript at the OLD budget looks like a
                # bookkeeping error rather than the deliberate fallback it is.
                **({'turn_cap_rerun_failed': turn_cap_rerun_failed} if turn_cap_rerun_failed else {}),
                # WHICH BACKEND produced the verdicts in this file. Written
                # ALWAYS, ``null`` included: a backend that sent no stamp and a writer that
                # records none are different facts, and only the first is a reason to go look
                # at the server. Read straight off THIS scenario's finalize response — one
                # writer, no second capture at /threads.
                'backend_build': _record_backend_stamp(result.finalize),
                # WHY the backend did not certify this row's world, when it did not. Present ONLY
                # on such a row — an absent key is a row
                # whose world was fine, which is the common case and needs no sentence.
                **({'untrustworthy_world': reason} if (reason := _record_untrustworthy(scenario, result)) else {}),
                **(run_context or {}),
            },
            # The CONVERSATION FACTS of the /finalize payload (``verdicts.RECORD_KEYS``: how the dialog ended,
            # the simulated user's integrity, …) — what the backend's re-score reads back from this file. The
            # VERDICTS are not here: they are in ``verdicts.json``, written below by its one writer, and this
            # file is never rewritten when a verdict changes.
            'finalize_record': verdicts.finalize_record(per),
        },
    )
    # The server returns the normalized {entities: {...}} snapshot.
    _write_json(scenario_out / 'final_world_state.json', result.finalize.final_world_state or {})
    # NOT ``or {}``: the file is always written (it is in ``REQUIRED_ARTIFACT_FILES``), but an
    # unread tool channel is recorded as a literal ``null``. Writing ``{}`` would tell every
    # later reader — the re-score lane above all — that this run measured a trail and found it
    # empty, which is the one claim a run with no channel cannot make. The backend's re-score path
    # already asks the question this answers (is ``tool_events`` unsupplied); the backend's upload
    # loader must admit a top-level ``null`` for this filename; and the plugin's post-run report builder
    # (its ``tool_events.json`` read) must
    # tell a parsed ``null`` from an absent file — it reads both as "the scenario did not
    # finalize" and publishes ``0 tool calls`` in the customer-facing artifact. All three halves must agree.
    _write_json(scenario_out / 'tool_events.json', result.tool_events)
    # The forensic twin of the block above, written ONLY for a hop — the one file a customer
    # opens to see what the component was given and what it returned, and the file the setup
    # skill names. Standalone (no re-score role), so it
    # restates the inputs — ``initial_agent_context`` as recorded — rather than
    # pointing at another file, and it holds the component's raw outputs where the transcript holds
    # their wire projection.
    #
    # EXACTLY these five keys: ``scenario_metadata`` or ``policy_evaluation_results`` here would
    # put the JUDGES' GROUND TRUTH on the customer's disk for ONE shape only. A dialog artifact carries neither — its ``metadata``
    # block holds the scenario NAME and run identity and nothing of the judge's material — and a
    # hop must not be the one shape that leaks. The verdicts are already in ``verdicts.json``
    # and ``run.json``, so nothing is lost. No ``expected`` either: authored expectations have
    # never had a producer, and ground truth belongs to the eval set, not to a run artifact.
    if result.pipeline_io is not None:
        raw_outputs = result.pipeline_raw_outputs if result.pipeline_raw_outputs is not None else result.pipeline_io['outputs']
        _write_json(scenario_out / 'pipeline_io.json', {'inputs': result.initial_agent_context, **result.pipeline_io, 'outputs': raw_outputs})
    # LAST: how this attempt was judged, from the same /finalize response — its presence says the attempt's
    # artifacts are complete.
    verdicts.write_initial(
        scenario_out,
        attempt=verdicts.attempt_of(scenario_out, scenario=scenario, thread_id=result.thread_id, run_context=run_context),
        identity=run_context or {},
        per=per,
        ended_early=result.ended_early,
        turn_failure=result.turn_failure,
        cut_by_harness=_row_outcome(result) == 'incomplete',
        finalize_job_id=result.finalize_job_id,
    )


#: How much bigger the second attempt's turn budget is when ``--rerun-on-turn-cap`` fires.
#: Policy, not a knob: the first attempt already proved the budget was too small, and the only
#: question is whether DOUBLE is enough to reach the target. A finer dial would be an operator
#: guessing at a number the run can measure for itself — if double is still not enough, the
#: second attempt records ``turn_cap_exhausted`` too and the honest fix is ``--max-turns``.
TURN_CAP_RERUN_MULTIPLIER = 2


@dataclass(frozen=True, slots=True)
class _TurnCapRerunProvenance:
    """The record of a ceiling-hit re-run, persisted into the re-run's own artifacts.

    Both attempts are recorded — this block on the fresh one, the superseded attempt's
    artifacts under :data:`PREVIOUS_ATTEMPT_DIRNAME` — so "the first run of this scenario ran
    out of turns and here is what it did" is answerable from the folder rather than from an
    operator's memory of having re-run it by hand.
    """

    previous_thread_id: str
    previous_max_turns: int
    max_turns: int

    def to_artifact(self) -> dict[str, Any]:
        return {
            'previous_thread_id': self.previous_thread_id,
            'previous_max_turns': self.previous_max_turns,
            'max_turns': self.max_turns,
        }


def _log_turn_cap_rerun_declined(scenario_id: str, target: Literal['complete', 'unknown']) -> None:
    """SAID, not silently skipped.

    An operator who passed ``--rerun-on-turn-cap`` and sees a capped scenario go un-re-run is
    owed the reason, and the two are different situations: "it got there anyway" needs nothing,
    while "the backend told us nothing" means the feature cannot work against this box.
    """
    logger.info(
        'scenario %s hit its turn cap but was NOT re-run: %s.',
        scenario_id,
        'the user simulator reported its target already met'
        if target == 'complete'
        else 'no usable target verdict came back from finalize (an older backend, or a finalize that '
        'never got that far), and a re-run must not be spent on a guess',
    )


def _announce_turn_cap_rerun(scenario_id: str, rerun: _TurnCapRerunProvenance) -> None:
    """Say that this scenario is about to be run a SECOND time, and what it cost."""
    logger.warning(
        'scenario %s hit its turn cap (%d) with its target still unmet; re-running once at %d turns. Both attempts are recorded — this one under %s/.',
        scenario_id,
        rerun.previous_max_turns,
        rerun.max_turns,
        PREVIOUS_ATTEMPT_DIRNAME,
    )
    print(
        f'[{_ts()}] [turn-cap-rerun] {scenario_id}: ran out of turns at {rerun.previous_max_turns} with the '
        f'target unmet; re-running once at {rerun.max_turns}. A budget artifact is not an agent verdict.',
        file=sys.stderr,
    )


def _target_status(result: ThreadResult) -> Literal['incomplete', 'complete', 'unknown']:
    """What the BACKEND says about this conversation reaching its scenario target.

    Read off the finalize payload's ``per_turn_user_step_info`` — the user simulator's own
    per-turn verdict — never re-derived from the transcript. The highest turn's entry is the
    last word on it.

    THREE-valued, because the two ways of not being ``incomplete`` need different handling and
    a boolean would collapse them into one silent ``False``: an operator who passed
    ``--rerun-on-turn-cap`` and saw no re-run could not tell "it reached its target anyway"
    (nothing to do) from "this backend reports no target verdict at all" (the feature cannot
    work here). Neither spends the attempt — a re-run must never be spent on a guess — but only
    one of them is fine.
    """
    info = getattr(result.finalize, 'per_turn_user_step_info', None)
    if not isinstance(info, dict) or not info:
        return 'unknown'
    try:
        last = info[max(info, key=int)]
    except (TypeError, ValueError):
        return 'unknown'
    if not isinstance(last, dict) or not isinstance(last.get('target_complete'), bool):
        return 'unknown'
    return 'incomplete' if last['target_complete'] is False else 'complete'


@dataclass(frozen=True, slots=True)
class _PriorAttempt:
    """What a scenario folder holds after making room for a fresh attempt.

    THREE states, because two cannot express the one that matters. ``preserved`` names the
    directory holding a readable earlier attempt; ``stale`` names the top-level artifacts that
    could NOT be moved or removed — a folder that is about to be described by two different
    attempts at once, which is the condition the whole hygiene exists to prevent and the one
    that must never be reported as either "preserved" or "clean".
    """

    preserved: str | None = None
    stale: tuple[str, ...] = ()

    def without(self, names: Iterable[str]) -> _PriorAttempt:
        """This record with *names* dropped from ``stale`` — the files that are no longer stale.

        ``stale`` is computed BEFORE the attempt runs, and a successful attempt then OVERWRITES
        exactly the artifact files it names. Emitting the note unfiltered would put a valid
        ``pass_rate`` beside "read them as the earlier attempt's, not as this one's" — telling a
        reader, and ``/analyze-simulation``'s critic, to discount a perfectly good measurement.
        The writer that rewrites a file is the one place that knows it did, so the filtering
        happens there rather than at each call site.
        """
        dropped = set(names)
        return dataclasses_replace(self, stale=tuple(name for name in self.stale if name not in dropped))

    def to_artifact(self) -> dict[str, Any] | None:
        """The hygiene note for this attempt's artifacts, or ``None`` when there is nothing to say."""
        if not self.stale:
            return None
        return {
            'stale_artifacts': list(self.stale),
            'detail': (
                'these files are from an EARLIER attempt and could not be moved aside or removed, so this '
                "scenario folder describes two attempts at once. Read them as the earlier attempt's, not as "
                "this one's, and fix the permissions/disk condition that blocked the cleanup."
            ),
        }


@dataclass(frozen=True, slots=True)
class _AttemptProvenance:
    """Where ONE attempt sits in its scenario folder's history.

    The three facts travel together through :func:`_run_one` and land in whichever description
    file the folder ends up carrying, so they are built in ONE place: a reader who opens
    ``transcript.json`` and a reader who opens ``error.json`` must find the same answers, and a
    key added to one site and not the other is how they stop agreeing.
    """

    prior: _PriorAttempt = _PriorAttempt()
    turn_cap_rerun: _TurnCapRerunProvenance | None = None

    def without_stale(self, names: Iterable[str]) -> _AttemptProvenance:
        """This provenance with *names* no longer counted stale — see :meth:`_PriorAttempt.without`."""
        return dataclasses_replace(self, prior=self.prior.without(names))

    def to_metadata(self) -> dict[str, Any]:
        """The artifact keys for this attempt — each ABSENT when it has nothing to say.

        Absent rather than null throughout: an always-present marker would make every ordinary
        first run look like a re-run of something.
        """
        hygiene = self.prior.to_artifact()
        return {
            # Present ONLY when this scenario folder also holds the attempt this one superseded
            # (``previous_attempt/``). It is what makes "there was an earlier run of this
            # scenario, and here it is" readable from the artifact rather than from a listing
            # nobody takes — and in error.json it is doubly load-bearing, because that is the
            # file a reader opens when the folder looks empty: without it, the surviving
            # measurement is a directory nobody thinks to look in.
            **({'previous_attempt': self.prior.preserved} if self.prior.preserved else {}),
            # Present ONLY when the folder could not be cleared for this attempt, naming the
            # earlier attempt's files that are still sitting beside these ones. A reader who
            # cannot see this reads a mixed folder as one run.
            **({'artifact_hygiene': hygiene} if hygiene else {}),
            # Present ONLY on the second attempt of a ceiling-hit re-run: the thread that ran
            # out of turns, its budget, and the raised one. Without it, a re-run at a different
            # budget is indistinguishable from a first run — and the two are not comparable
            # measurements. On a FAILURE it is what says the failure was the runner's own second
            # attempt, not a first run that failed.
            **({'turn_cap_rerun': self.turn_cap_rerun.to_artifact()} if self.turn_cap_rerun else {}),
        }


@dataclass(frozen=True, slots=True)
class _SupersededAttempt:
    """A finished measurement a ceiling re-run is betting against, with the stamps that describe IT.

    The stamps travel WITH the result because the re-run overwrites the loop's own: ``started_at``
    is re-captured per attempt and ``attempt_max_turns`` is doubled, so restoring from the loop's
    current values would stamp attempt 1's transcript with attempt 2's clock.
    """

    result: ThreadResult
    started_at: str
    max_turns: int


def _restore_superseded_attempt(
    cfg: _RunConfig,
    settings: Settings,
    scenario_out: Path,
    *,
    scenario_id: str,
    superseded: _SupersededAttempt,
    dataset_name: str | None,
    run_name: str | None,
    turn_cap_rerun: _TurnCapRerunProvenance | None,
    failure: BaseException,
    ledger: dict[str, dict[str, Any]] | None,
) -> ThreadResult:
    """Give back the attempt a FAILED ceiling re-run was trying to improve on.

    The re-run is an optional bet the runner makes with a measurement already in hand, and
    losing the bet must not cost the stake. Writing ``error.json`` instead would drop this row
    out of ``graded_scenarios``, ``ended_early_scenarios`` and ``measurement_integrity`` alike
    and publish "6/7 succeeded, 1 FAILED" for a run in which 7 of 7 ran and one ended at its
    cap — with the surviving measurement sitting on disk, unread and unuploaded.

    The folder is rewritten from the HELD result rather than by moving files back: the artifacts
    are a pure function of it, so this cannot produce a half-restored set, and it is the only
    way to add the ``turn_cap_rerun_failed`` note that says why this row is the earlier attempt.
    ``previous_attempt/`` is then removed — it held this very attempt, and leaving a duplicate
    of the top level behind would invite a reader to diff two identical directories.

    NO ``error.json``: the folder describes exactly one attempt, and that attempt succeeded.
    The failed re-run is recorded in the transcript's own metadata (and in the ledger the run
    summary publishes), which is where a reader of this row will actually be.
    """
    failed = {'exception': type(failure).__name__, 'error': str(failure) or repr(failure)}
    _write_success_artifacts(
        scenario_out,
        scenario=scenario_id,
        result=superseded.result,
        # THIS attempt's stamps, not the loop's — the loop has moved on to the re-run that just
        # failed (see _SupersededAttempt).
        run_context=_run_context(cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=superseded.started_at, max_turns=superseded.max_turns),
        # No ``prior``: ``previous_attempt/`` is removed just below, so naming it here would
        # point a reader at a directory this function is about to delete.
        attempt=_AttemptProvenance(turn_cap_rerun=turn_cap_rerun),
        turn_cap_rerun_failed=failed,
    )
    shutil.rmtree(scenario_out / PREVIOUS_ATTEMPT_DIRNAME, ignore_errors=True)
    if ledger is not None and scenario_id in ledger:
        ledger[scenario_id] = {**ledger[scenario_id], 'outcome': _RERUN_RESTORED_OUTCOME, 'rerun_failure': failed}
    logger.warning(
        'scenario %s: the turn-cap re-run failed (%s: %s). The attempt it superseded is a real measurement and has '
        'been restored as this row — the re-run is an optional improvement, not a precondition.',
        scenario_id,
        failed['exception'],
        failed['error'],
    )
    print(
        f'[{_ts()}] [turn-cap-rerun] {scenario_id}: the re-run FAILED ({failed["exception"]}); keeping the earlier '
        f'attempt, which ran to its turn cap. This row is a measurement, not a failure.',
        file=sys.stderr,
    )
    return superseded.result


def _drop_stale_error_json(scenario_out: Path) -> list[str]:
    """Remove an earlier attempt's error record (``ERROR_RECORD_FILES``); returns the names that SURVIVED.

    Dropped rather than preserved: it carries no measurement, and this attempt's own outcome
    supersedes it either way. A stale one that CANNOT be removed is an invariant breach, not a
    detail — ``run_units`` files a folder as ``failed`` on the strength of an ``error.json``
    even when a transcript sits beside it, so a leftover turns this attempt's pass into a
    reported failure. Logged at WARNING (a DEBUG line goes to nobody: this package never calls
    ``basicConfig``, so the stdlib's lastResort handler prints at WARNING and above).
    """
    if (scenario_out / 'error.json').exists():
        # The failed attempt's verdicts.json (its status) is part of that error record and goes with it.
        try:
            verdicts.discard_with_error_record(scenario_out)
        except OSError:
            logger.warning('could not remove the stale verdicts.json beside error.json in %s; it belongs to an EARLIER attempt.', scenario_out, exc_info=True)
    for name in ERROR_RECORD_FILES:
        try:
            (scenario_out / name).unlink(missing_ok=True)
        except OSError:
            logger.warning(
                'could not remove the stale %s in %s. A leftover error record makes THIS attempt read as a '
                'failure even when it passes (an upload files the folder by which description file it carries), so this '
                "scenario's outcome on disk is not trustworthy until it is removed by hand.",
                name,
                scenario_out,
                exc_info=True,
            )
    return [name for name in ERROR_RECORD_FILES if (scenario_out / name).exists()]


def _drop_stale_partial_evidence(scenario_out: Path) -> list[str]:
    """Remove an earlier attempt's partial evidence; the names that SURVIVED.

    It explains the ``error.json`` dropped beside it and goes with it: left behind, it would sit
    next to this attempt's artifacts describing a conversation this attempt never had. The
    ``.tmp`` siblings a torn :func:`_write_json` can leave go as well.
    """
    survived: list[str] = []
    for name in (*PARTIAL_EVIDENCE_FILES, *(f'{name}.tmp' for name in PARTIAL_EVIDENCE_FILES)):
        path = scenario_out / name
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning('could not remove the stale %s in %s; it belongs to an EARLIER attempt, not this one.', name, scenario_out, exc_info=True)
        if path.exists():
            survived.append(name)
    return survived


def _unstage(staging: Path, scenario_out: Path, names: Iterable[str]) -> None:
    """Put back what a FAILED displacement had already moved into ``staging``.

    The top level is where this scenario's own writer will look, and a file stranded in a temp
    directory is lost to every reader. Anything that will not move back is left where it is and
    named as stale by the caller rather than deleted — destroying a measurement to tidy a folder
    is the defect the displacement exists to remove.
    """
    for name in names:
        source, target = staging / name, scenario_out / name
        if source.is_file() and not target.exists():
            try:
                source.replace(target)
            except OSError:
                logger.debug('could not restore %s from staging in %s', name, scenario_out, exc_info=True)
    shutil.rmtree(staging, ignore_errors=True)


def _recover_stranded_staging(scenario_out: Path) -> None:
    """Reclaim the artifacts a KILLED displacement left in a staging directory.

    The staging swap is atomic against a crash between the individual moves, but not against one
    DURING them: a Ctrl-C / OOM / SIGKILL after the first ``replace`` leaves part of the attempt
    in ``previous_attempt.<pid>.tmp/`` and part at the top level. Nothing else names that
    directory, so the next run would compute ``present`` without it and could publish a
    ``previous_attempt/`` holding ``final_world_state.json`` + ``tool_events.json`` and NOT
    ``transcript.json`` — the "keeps the two carrying no verdict, loses the one carrying it"
    outcome the staging design exists to make impossible. With everything staged and nothing left
    at the top, the only copy of a measurement would sit in a directory the artifacts never
    mention and ``plurai ingest-run`` would report as a nested run tree.

    So every stray is swept BEFORE this attempt looks at the folder: each artifact whose
    top-level name is FREE moves back up, rejoining ``present`` and being displaced properly.
    A name already taken at the top level belongs to a LATER attempt and wins — the stray copy
    is older, and the top level describes the current attempt by construction.

    Sweeping first is also what makes pid reuse safe: this process's own staging name is
    recovered here rather than deleted blind by the ``mkdir`` below.
    """
    for staging in sorted(scenario_out.glob(f'{PREVIOUS_ATTEMPT_DIRNAME}.*.tmp')):
        if not staging.is_dir():
            continue
        logger.warning(
            'scenario %s holds a staging directory (%s) from an interrupted run: its artifacts are being reclaimed '
            'before this attempt starts. A previous run was killed mid-displacement.',
            scenario_out.name,
            staging.name,
        )
        _unstage(staging, scenario_out, SCENARIO_ARTIFACT_FILES)


def _displace_prior_attempt(scenario_out: Path) -> _PriorAttempt:
    """Make room for a fresh attempt WITHOUT destroying the one it supersedes.

    The invariant this protects: the scenario folder's TOP LEVEL describes exactly
    one attempt, so a fresh ``error.json`` can never sit beside a stale ``transcript.json`` and
    make the documented failure sweep (``find … -name error.json``) report a now-passing
    scenario as failed. The superseded attempt goes aside, not away.

    Deleting it up front would make the folder's contents depend on an outcome that has not
    happened yet: a re-run that crashed, was killed, or was merely still in flight would leave
    NOTHING, and a measurement already taken would be gone.

    ATOMIC, via a staging directory and ONE rename. The obvious implementation — clear
    ``previous_attempt/`` then move three files into it — is interruptible at four points, and
    the damage is not symmetric: a Ctrl-C between the moves leaves a half-set, and the NEXT run
    then clears that directory wholesale and deletes the preserved ``transcript.json`` (the only
    one of the three carrying a verdict) while dutifully keeping the two that carry none. So the
    files are staged into ``previous_attempt.<pid>.tmp/`` first and swapped in with a single
    ``replace``: until that instant the previous attempt is untouched, and after it the swap has
    already happened.

    ``error.json`` is dropped rather than preserved, together with the partial evidence beside
    it, and a stale one that will not go is reported as ``stale`` — see
    :func:`_drop_stale_error_json` and :func:`_drop_stale_partial_evidence`.

    Never raises: a scenario must not fail because of artifact housekeeping. What it will not do
    is claim more than it did — every unremovable file is named in the returned record and, from
    there, in the artifacts.
    """
    stale: list[str] = _drop_stale_error_json(scenario_out)
    stale.extend(_drop_stale_partial_evidence(scenario_out))
    # BEFORE anything reads the folder: an interrupted run may have left part of an attempt in a
    # staging directory, and it has to rejoin ``present`` to be displaced whole.
    _recover_stranded_staging(scenario_out)

    present = [name for name in SCENARIO_ARTIFACT_FILES if (scenario_out / name).is_file()]
    keep = scenario_out / PREVIOUS_ATTEMPT_DIRNAME
    if not present:
        # Nothing to displace. A ``previous_attempt/`` is only a preserved measurement when it
        # actually HOLDS one — an empty directory (a cleanup that got that far, a copy that did
        # not) would otherwise be published as a surviving attempt that nothing can read.
        preserved = PREVIOUS_ATTEMPT_DIRNAME if any((keep / name).is_file() for name in SCENARIO_ARTIFACT_FILES) else None
        return _PriorAttempt(preserved=preserved, stale=tuple(stale))

    staging = scenario_out / f'{PREVIOUS_ATTEMPT_DIRNAME}.{os.getpid()}.tmp'
    try:
        # No pre-clear: the sweep above already reclaimed any stray under this name, so a
        # directory here now would be a genuine surprise and ``mkdir`` failing on it is the
        # honest outcome (it lands in the except below and is reported, not deleted blind).
        staging.mkdir(parents=True)
        for name in present:
            (scenario_out / name).replace(staging / name)
        # The previous attempt is replaced WHOLESALE, never merged: a directory holding
        # transcript.json from one attempt and tool_events.json from another describes a
        # conversation that never happened. Removing it is the last destructive act, and by now
        # this attempt's files are already staged, so an interruption here costs the older
        # attempt only — never the one being preserved.
        shutil.rmtree(keep, ignore_errors=True)
        staging.replace(keep)
    except OSError:
        _unstage(staging, scenario_out, present)
        survived = [name for name in SCENARIO_ARTIFACT_FILES if (scenario_out / name).exists()]
        stale.extend(survived)
        logger.warning(
            'could not preserve the previous attempt in %s. These files are still at the top level and belong to the '
            'EARLIER attempt: %s. This folder now describes two attempts at once — the artifacts written here name '
            'them so a reader is not left guessing.',
            scenario_out,
            ', '.join(survived) or '(none)',
            exc_info=True,
        )
        return _PriorAttempt(preserved=None, stale=tuple(stale))
    logger.info(
        'scenario %s already held an attempt; it is preserved in %s/ and this run starts fresh.',
        scenario_out.name,
        PREVIOUS_ATTEMPT_DIRNAME,
    )
    return _PriorAttempt(preserved=PREVIOUS_ATTEMPT_DIRNAME, stale=tuple(stale))


#: A tool_call ``response``'s well-known "list endpoint answered with nothing" shape — the one
#: :class:`~plurai._protocol.GraphRecursionWrap`'s own docstring names ("REMOTE-SIDE
#: empty mock ... ``{"results": []}``"). Checked ALONGSIDE a bare ``status``/``ok``/``success``
#: sibling key (a bookkeeping field, not a payload) so a response shaped exactly like the
#: payload carrying a ``content`` field (``{"status": "success", "content": "..."}``)
#: is never mistaken for this shape.
_EMPTY_CONTAINER_KEYS = ('results', 'items', 'data', 'records', 'rows')
_BOOKKEEPING_RESPONSE_KEYS = ('status', 'ok', 'success', 'error')


def _tool_response_is_empty(response: Any) -> bool:
    """Shallow "no payload" check for one tool_call event's ``response`` — never a specific
    tool's schema, so it reads identically whatever mock produced the event.

    ``None``, a blank/whitespace string, and an empty list/dict are empty. A JSON-encoded
    string is decoded one level before the same check applies (every captured ``response`` in
    ``tool_events.partial.json`` is a JSON string, e.g. ``'{"status": "success", "content":
    "Deleted 1 record, 8 remaining"}'``). A dict whose only non-bookkeeping key is one of
    :data:`_EMPTY_CONTAINER_KEYS` mapped to an empty list is ALSO empty — the case-B shape.
    Anything else (a real field, a non-empty nested container, a bare scalar) is data.
    """
    if isinstance(response, str):
        text = response.strip()
        if not text:
            return True
        try:
            response = json.loads(text)
        except (ValueError, TypeError):
            return False  # a non-empty, non-JSON string IS data
    if response is None:
        return True
    if isinstance(response, (list, tuple)):
        return len(response) == 0
    if isinstance(response, dict):
        if not response:
            return True
        meaningful = {k: v for k, v in response.items() if k not in _BOOKKEEPING_RESPONSE_KEYS}
        if not meaningful:
            return True
        if len(meaningful) == 1:
            ((key, value),) = meaningful.items()
            if key in _EMPTY_CONTAINER_KEYS and isinstance(value, (list, tuple)) and len(value) == 0:
                return True
        return False
    return False  # a bare scalar (number/bool) is data


#: :func:`_pre_crash_tool_activity`'s three-valued read of the crashed attempt's own tool
#: events — see its docstring. Named to match ``pre_crash_tool_activity`` on the wire.
PreCrashToolActivity = Literal['real_data', 'empty_or_none', 'unknown']


def _pre_crash_tool_activity(progress: ScenarioProgress | None) -> PreCrashToolActivity:
    """Did the crashed attempt's own recorded tool calls include at least one non-empty response?

    Reads ONLY :meth:`ScenarioProgress.tool_events` — the SAME merged log
    :func:`_write_partial_evidence` writes to ``tool_events.partial.json`` — never a stack
    frame, never the raising exception's class name, so this applies identically to any future
    recursion/step-limit family a detector is added for (:data:`_RECURSION_LIMIT_DETECTORS` in
    ``_protocol.py``). Turn-scoped evidence of ACTIVITY, not a diagnosis of which of
    ``GraphRecursionWrap``'s own A/B/C causes this is — a caller must not read ``'real_data'``
    as "the agent looped, not the harness"; it is read only as "this run's own attempt was not
    silent before it crashed".

    ``'unknown'`` when there is no tool-events channel to read at all (``progress`` is
    ``None``, its ``tool_events()`` returns ``None`` — no channel was read and nothing was
    recovered — or reading it RAISES) — a DIFFERENT fact from ``'empty_or_none'``, which means
    the channel WAS read and every call recorded in it came back with nothing.

    Best-effort, like :func:`_write_partial_evidence` right beside it: this is the FAILURE path
    (an exception is already being written to ``error.json``), and a merge that raises — a
    duck-typed test double, a real recovery bug — must not cost the record it would decorate.
    ``tool_events()`` is documented elsewhere as a call that CAN raise, so this reads it the
    same guarded way.
    """
    if progress is None:
        return 'unknown'
    try:
        tool_events = progress.tool_events()
    except Exception:  # noqa: BLE001 — the failure path; see the docstring above
        return 'unknown'
    if not isinstance(tool_events, dict):
        return 'unknown'
    events = tool_events.get('events')
    if not isinstance(events, list):
        return 'unknown'
    calls = [e for e in events if isinstance(e, dict) and e.get('type') == 'tool_call']
    if not calls:
        return 'empty_or_none'
    if any(not _tool_response_is_empty(call.get('response')) for call in calls):
        return 'real_data'
    return 'empty_or_none'


def _write_error(
    scenario_out: Path,
    *,
    scenario: str,
    thread_id: str,
    exc: BaseException,
    redrive: RedriveProvenance | None = None,
    attempt: _AttemptProvenance | None = None,
    progress: ScenarioProgress | None = None,
    turns_reached: Sequence[Mapping[str, Any]],
    world_slot: int | None = None,
    run_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``error.json``; returns the record so the caller's stderr line can share it.

    Then the attempt's ``verdicts.json`` (:func:`plurai.verdicts.write_initial`, ``graded: false``): its STATUS,
    computed once from this record, so no reader re-classifies ``error.json``. ``run_context`` is the invocation's
    identity block (:func:`_run_context`) — which trial, which eval / scenario set.

    ``world_slot`` is the world copy the failed attempt leased (``world_state.slots``), so a dead
    copy reads as the failures sharing one slot; absent with one shared world.

    ``turns_reached`` (required, so no failure exit can drop it) is the attempt's completed turns:
    their count rides ``error.json`` as an int, and the turns themselves land in
    ``ERROR_DIALOG_FILENAME`` — unscored evidence, never a transcript (see run_artifacts).

    ``str(exc) or repr(exc)``: httpx transport errors (ReadTimeout et al.) are routinely
    raised with NO message args, so bare ``str()`` would produce ``{"exception": "ReadTimeout",
    "error": ""}`` — an artifact that names a timeout but not which endpoint or URL. The request URL is captured for the same reason, best-effort.

    ``progress`` is the SAME object the caller passes to ``_write_partial_evidence`` right
    after this call — read here, ahead of that write, to stamp ``pre_crash_tool_activity``
    (see :func:`_pre_crash_tool_activity`). Generic and always stamped, never gated on
    ``exc``'s class: a fact about this attempt's recorded tool events, not a fact this
    function decides is worth knowing only for one exception family.
    """
    scenario_out.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {'scenario': scenario, 'thread_id': thread_id, 'exception': type(exc).__name__, 'error': str(exc) or repr(exc)}
    record['turns_reached'] = len(turns_reached)
    if world_slot is not None:
        record['world_slot'] = world_slot
    # The TYPED classification, read off the raising class — never re-derived from the
    # message. Absent when the class declares none, because "unknown" and "not a harness
    # fault" are different answers and a default would merge them. Consumers key on this
    # to keep a simulation-budget expiry out of the agent's scorecard: the run report and
    # the suite critic both read a bare `result: "error"` as the agent crashing, and a
    # timeout carries no transcript for them to check that against.
    kind = getattr(type(exc), 'kind', None)
    if kind is not None:
        record['kind'] = kind
    if kind == ScenarioQuarantined.kind:
        # Noted HERE because this is the last frame holding the typed exception; the row status is
        # decided later from a scenario id and a ``None``. See :data:`_QUARANTINED_ROWS`.
        _run_ledgers().quarantined.add(scenario)
    if is_harness_owned(kind):
        _run_ledgers().harness_fault.add(scenario)
    # Additive, generic evidence — never a diagnosis (see the function's own docstring).
    # ``trials.py``'s carve-out and the backend's run-error routing are the only two readers
    # that give this field a verdict; every other consumer reads it as extra context.
    record['pre_crash_tool_activity'] = _pre_crash_tool_activity(progress)
    # ``envelope_type`` is what the budget was WAITING on, and it changes the remedy: a
    # stalled user_turn is the provider behind the agent, an init stall is simulate.py
    # being slow to CONSTRUCT the agent. Carried as a field so a consumer never has to
    # regex it back out of the message prose.
    envelope_type = getattr(exc, 'envelope_type', None)
    if envelope_type is not None:
        record['envelope_type'] = envelope_type
    # Which side's IPC line overran (``HarnessIpcOverrun.direction``: ``sent`` / ``received``) —
    # beside ``envelope_type``, which names the same ``init`` for an oversized Init and its reply.
    if isinstance(exc, HarnessIpcOverrun):
        record['direction'] = exc.direction
    # The child's exit code, whose SIGN is the classification (negative ⇒ ``-signal``, i.e.
    # reaped rather than exited). Carried as a field so a consumer never has to regex it
    # back out of the message — the same rule ``kind`` and ``budget_seconds`` follow.
    returncode = getattr(exc, 'returncode', None)
    if returncode is not None:
        record['returncode'] = returncode
    # The budget that was actually in force, so a report can name the knob to raise
    # instead of just the wall that was hit.
    budget_seconds = getattr(exc, 'budget_seconds', None)
    if budget_seconds is not None:
        record['budget_seconds'] = budget_seconds
    if getattr(exc, 'timed_out', False):
        record['timed_out'] = True
    # WHEN in the scenario's lifecycle the raise fired — 'init' or 'finalize' — for a kind whose
    # remedy depends on the phase (e.g. harness_mock_unreachable: a mockless-posture suggestion is
    # a live possibility at init, a non-sequitur at finalize, since the agent already dialed the
    # mock by then). Absent on a raiser with no phase-dependent remedy — a consumer reading `error.json` must treat that absence as
    # "unknown," never as a synonym for either phase.
    raised_at = getattr(exc, 'raised_at', None)
    if raised_at is not None:
        record['raised_at'] = raised_at
    # A mock-runtime fault's class and the runtime's own cause, typed fields like ``budget_seconds``: the
    # remedy (re-run, or the backend's to fix) is read off ``transient``, never out of the message.
    if isinstance(exc, MockRuntimeFaultRaised):
        record['transient'] = exc.transient
        record['cause'] = exc.cause
        # Why the mock never answered (``transport`` ⇒ re-run; any other names what to fix); ``null`` when the runtime answered with a fault.
        record['route_cause'] = exc.route_cause
    # The orchestration server's own HTTP failure (``kind='backend_http'``) or credit refusal (``kind='backend_credit'``): its status,
    # when an HTTP answer carried one (a finalize or rescore job's refusal has none), and whether a re-run can answer.
    if isinstance(exc, ServerHTTPError | CreditRefused):
        if exc.status is not None:
            record['status'] = exc.status
        record['transient'] = exc.transient
    # An answer that did not come from Plurai (a proxy or firewall): the typed record beside the unchanged ``kind`` and ``status``.
    if isinstance(exc, NetworkRefused):
        record['network_refusal'] = exc.refusal.to_record()
    # A credit refusal (from a 402 or a finalize job): the kind and, for the free tier, the reset time, as typed fields.
    if (credit := credit_block_of(exc)) is not None:
        record['credit'] = credit.to_record()
    # A capacity refusal (a hosted mock's fault): the backend's block, so a reader renders the stop from the record alone.
    if (capacity := capacity_block_of(exc)) is not None:
        record['capacity'] = capacity.to_record()
    # The backend took the connection and never answered in time: the customer's line says so without the transport's text.
    if isinstance(exc, NON_OUTAGE_TIMEOUTS):
        record['backend_no_answer'] = True
    # A transient failure the retry policy did not repeat (a create that may already have been forwarded) and that spent no
    # budget: no outage, and the backend may have taken the request. The customer's line names neither the status nor the
    # transport's text.
    elif is_transient_for_idempotent(exc) and unavailable_block_of(exc) is None and not is_name_not_found(exc):
        record['request_not_repeated'] = True
    # An unavailable stop: the cause alone, in place of the transport's own text.
    if (unavailable := unavailable_block_of(exc)) is not None:
        record['unavailable'] = unavailable.to_record()
    # A re-drive that ALSO failed is the loudest case there is — record the
    # provenance here too, so the failure report says "this was the second
    # attempt, and here is what killed the first".
    if redrive is not None:
        record['redrive'] = redrive.to_artifact()
    # Where this attempt sits in the folder's history — the SAME block transcript.json carries,
    # and doubly worth having here, because error.json is the file a reader opens when the
    # folder looks empty. See _AttemptProvenance.to_metadata.
    if attempt is not None:
        record.update(attempt.to_metadata())
    try:
        # ``.request`` is a RAISING property on an httpx error whose request was never bound —
        # and getattr's default only swallows AttributeError. Same reasoning as the ``.text``
        # read below: this is the FAILURE path, so a raise here would abort the whole gather
        # and destroy the sibling scenarios' results along with this record.
        request = getattr(exc, 'request', None)
    except Exception:  # noqa: BLE001
        request = None
    url = getattr(request, 'url', None)
    if url is not None:
        record['request_url'] = str(url)
    # httpx's raise_for_status message carries the status line and the request URL but NEVER
    # the response BODY, so the backend's message (which names the offending field, and on a
    # scenario-set miss the backend that answered) is persisted here as its customer-safe
    # description with the error id to quote, never as the raw body. Best-effort duck-typing:
    # any exception carrying a readable response is described, and a response that cannot be
    # read contributes nothing.
    response = getattr(exc, 'response', None)
    body = response_json(response)
    if description := describe_body(body, limit=2000, empty=''):
        record['response_body'] = description
    if error_id := error_id_of(body):
        record['error_id'] = error_id
    # WHAT THIS FAILURE INTERRUPTED. A scenario ceiling that expires while the finalize poll is
    # sleeping off a network outage raises ScenarioTimeout with the transport failure as its
    # context — without the chain the record says only "raise --scenario-timeout", with no
    # trace of the outage anywhere. The chain is what distinguishes "this scenario is legitimately long"
    # from "the box went away and the wall arrived first", and those need opposite responses.
    # ``__cause__`` (explicit ``raise ... from``) preferred over ``__context__`` (implicit),
    # one level only: the immediate cause is the diagnosis, and walking further just copies a
    # traceback into JSON.
    underlying = exc.__cause__ or exc.__context__
    if underlying is not None:
        record['caused_by'] = {'exception': type(underlying).__name__, 'error': str(underlying) or repr(underlying)}
    # BEFORE error.json, so error.json is the error record's last write (the partial evidence the
    # caller writes after it, ``_write_partial_evidence``, never touches either file). Best-effort:
    # this is the failure path, and a dialog that cannot be written must never cost the record itself.
    if turns_reached:
        try:
            _write_json(scenario_out / ERROR_DIALOG_FILENAME, {'turns': list(turns_reached)})
        except OSError:
            logger.warning('could not write %s in %s; error.json still records turns_reached', ERROR_DIALOG_FILENAME, scenario_out, exc_info=True)
    _write_json(scenario_out / 'error.json', record)
    try:
        verdicts.write_initial(
            scenario_out,
            attempt=verdicts.attempt_of(scenario_out, scenario=scenario, thread_id=thread_id, run_context=run_context),
            identity=run_context or {},
            error=record,
        )
    except (OSError, verdicts.VerdictsError):
        # The failure path: a raise here would abort the gather and the sibling scenarios with it.
        logger.warning('could not write verdicts.json in %s; error.json still records the failure', scenario_out, exc_info=True)
    return record


def _partial_transcript(progress: ScenarioProgress, *, scenario: str) -> dict[str, Any]:
    """The conversation of a failed attempt, as far as it got.

    A SUBSET of :func:`_write_success_artifacts`'s keys — the conversation, the injected inputs
    and the resume seed — spelled the same, so :func:`plurai.run_artifacts.load_transcript`
    reads it. Left out: everything a finished run alone can say (``user_turn_cycles``, the
    finalize payload's step info and verdicts, the restore-fidelity report, the run-context and
    provenance metadata). Nothing graded rides here: the attempt produced no verdict worth
    trusting, and ``error.json`` says why.
    """
    return {
        'transcript_schema': TRANSCRIPT_SCHEMA,
        'user_messages': [{'type': LC_MESSAGE_TYPES['user'], 'content': m} for m in progress.user_messages],
        'chatbot_messages': progress.chatbot_full_messages or [{'type': LC_MESSAGE_TYPES['agent'], 'content': m} for m in progress.chatbot_messages],
        'turns': progress.transcript_turns,
        'turn_count': len(progress.transcript_turns),
        'injected_inputs': {
            'initial_world_state': progress.initial_world_state,
            'initial_agent_context': progress.initial_agent_context,
            'post_seed_world_state': progress.post_seed_world_state,
            'resume_seed_present': progress.resume_seed is not None,
        },
        **({'resume_seed': progress.resume_seed} if progress.resume_seed else {}),
        'metadata': {
            'scenario': scenario,
            'thread_id': progress.thread_id,
            # Whether the partial tool events include the mock's own log. Before the child
            # finalized they hold only the calls recovered from the agent's side, and an absent
            # mock call there is not evidence that none was made.
            'mock_event_log_read': progress.mock_log_read,
        },
    }


def _write_partial_evidence(scenario_out: Path, *, scenario: str, progress: ScenarioProgress | None) -> None:
    """Write what a failed dialog attempt had produced beside its ``error.json``. Never raises.

    Runs AFTER ``error.json`` is written, and every failure here is logged and swallowed: this is
    the failure path, and evidence that cannot be written must never replace the error it
    explains. Each file is written on its own, so a tool-event merge that fails still leaves the
    conversation. ``None`` means the attempt failed before a dialog began, and nothing is written.
    Why the names differ from the success artifacts: :data:`plurai.run_artifacts.PARTIAL_EVIDENCE_FILES`.
    """
    if progress is None:
        return
    writes: tuple[tuple[str, Callable[[], Any]], ...] = (
        (PARTIAL_TRANSCRIPT_FILE, lambda: _partial_transcript(progress, scenario=scenario)),
        (PARTIAL_TOOL_EVENTS_FILE, lambda: progress.tool_events()),  # noqa: PLW0108 — even the attribute read happens inside the guard
    )
    for name, build in writes:
        try:
            _write_json(scenario_out / name, build())
        except Exception:  # noqa: BLE001 — see the docstring
            logger.warning('could not write %s for scenario %s; error.json is unaffected.', name, scenario, exc_info=True)
            with suppress(OSError):
                (scenario_out / f'{name}.tmp').unlink(missing_ok=True)


def _ts() -> str:
    return time.strftime('%H:%M:%S')


def _row_label(cfg: _RunConfig, scenario_id: str) -> str:
    """How a row's progress lines name it: the scenario id, and in a ``--trials N`` run its trial too (``<id>#t2``), since
    every trial runs the same ids side by side and their lines interleave."""
    return f'{scenario_id}#t{cfg.trial.index}' if cfg.trial is not None and cfg.trial.count > 1 else scenario_id


def _fail_line(scenario_id: str, record: dict[str, Any]) -> str:
    """The stderr failure line, sharing error.json's enrichment.

    A bare ``[FAIL] sc: HTTPStatusError: Client error '400 Bad Request' ...`` line hides
    the backend's actual explanation (the FastAPI ``detail`` naming the offending field —
    and on a scenario-set miss, the answering host) inside error.json, where finding it
    takes knowing to open the file. Echo the useful parts inline.
    """
    # A simulation fault is labelled AS one on the line an operator actually reads.
    # Unlabelled, the runner log's only account of a blown budget is an exception name
    # beside a wall of prose, and both the report and the suite critic downstream go on
    # to describe it as the agent failing.
    if record.get('kind') == ScenarioQuarantined.kind:
        # Its own label, ahead of the harness branch and separate from it. A suite that DECLINED
        # to take a measurement it could not trust is not our infrastructure breaking, and it is
        # not the agent failing either — it is a scenario that needs REPAIR, and a reader who
        # cannot see that in the run log goes looking for an agent bug that is not there.
        return f'[QUARANTINED] {scenario_id}: the SUITE refused to measure this scenario — {record["error"]}'
    if isinstance(record.get('credit'), dict):
        # Short on purpose: the kind's sentence is printed once, when the run ends.
        code = str(record['credit'].get('code'))
        return f'[{_stop_label(code)}] {scenario_id}: {_stop_cause(code)}; this scenario was not run'
    if isinstance(record.get('capacity'), dict) or isinstance(record.get('unavailable'), dict):
        # As a credit refusal: short, the capacity or unavailable text is printed once when the run ends. A scenario whose
        # thread was created began, and did not finish.
        began = bool(record.get('turns_reached')) or record.get('thread_id') not in (None, '', 'unknown')
        outcome = 'completed' if began else 'run'
        return f'[NOT RUN] {scenario_id}: {BACKEND_REFUSED_PHRASE}; this scenario was not {outcome}'
    if is_harness_owned(record.get('kind')):
        # Not every harness fault is a budget: a reaped child (harness_child_exited) is a
        # crash, and labelling it [TIMEOUT] would send an operator hunting a wall nothing hit.
        if record.get('timed_out'):
            return f'[TIMEOUT] {scenario_id}: SIMULATION budget, not an agent verdict — {record["error"]}'
        return f'[HARNESS] {scenario_id}: SIMULATION fault, not an agent verdict — {record["error"]}'
    if record.get('backend_no_answer'):
        # The backend took the connection and did not answer: no httpx text, status or URL in the customer's line (error.json keeps them).
        return f'[FAIL] {scenario_id}: the backend did not answer in time; this scenario failed'
    if record.get('request_not_repeated'):
        return f'[FAIL] {scenario_id}: the backend may not have taken this request; this scenario failed'
    line = f'[FAIL] {scenario_id}: {record["exception"]}: {record["error"]}'
    if record.get('request_url'):
        line += f' (request: {record["request_url"]})'
    body = record.get('response_body')
    if isinstance(body, str) and body.strip():
        line += f'\n       server said: {body[:500]}'
    return line


#: How often a still-running scenario says so. Long enough that a 7-scenario gather does not
#: become a wall of text, short enough that a wedged run is distinguishable from a working one
#: well inside a coffee break: without it a run's ONLY per-scenario output is the completion
#: line, so a healthy concurrent run and a hung one look identical for minutes at a time (and,
#: with block-buffered stdout, for the whole run).
_HEARTBEAT_INTERVAL_SECONDS = 120.0


async def _heartbeat(scenario_id: str, started: float, interval: float = _HEARTBEAT_INTERVAL_SECONDS) -> None:
    """Say that ``scenario_id`` is still going, every ``interval`` seconds, until cancelled.

    Elapsed time only. The turn cursor lives inside the orchestrator's loop and reaching it from
    here would be a second channel for a number the transcript already owns; what an operator
    watching a silent terminal needs first is whether anything is alive at all.

    A PROGRESS LINE MUST NEVER BECOME THE ROW'S OUTCOME. ``print`` is the only fallible statement
    in this loop and it is a likely one: these runs are always piped (the plugin tees them to a
    log), so a reader that goes away mid-run raises ``BrokenPipeError`` here — and this task's
    exception would otherwise replace a finished scenario's RESULT at the join below. It stops
    beating instead of retrying, because a stream that failed once will fail every 120 seconds
    for the rest of the run; the scenario itself is untouched either way.
    """
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - started
        try:
            print(f'[{_ts()}] [run]  {scenario_id}: still running ({elapsed / 60:.0f}m elapsed)')
        except Exception:  # noqa: BLE001 — see above: reporting progress may not decide a scenario's fate
            logger.debug('heartbeat for %s could not write to stdout; progress lines stop here.', scenario_id, exc_info=True)
            return


def _row_result(scenario_id: str, outcome: ThreadResult | None | BaseException) -> ThreadResult | None:
    """One settled ``asyncio.gather`` slot as a row outcome — an exception becomes a failed row.

    The gather runs with ``return_exceptions=True`` so one row cannot abandon its siblings and
    the run summary (see the call site). That makes the slot three-valued, and the exception case
    must be REPORTED rather than merely counted: ``_run_one`` writes ``error.json`` for every
    failure it handles itself, so a row arriving here as an exception is one whose failure
    happened outside that handling and has no artifact naming it.

    A ``BaseException`` that is not an ``Exception`` — cancellation, ``KeyboardInterrupt`` — is
    re-raised untouched. That is the RUN ending, not a row failing, and swallowing it into a
    ``failed`` count would report an interrupted run as a measurement of the agent.
    """
    if not isinstance(outcome, BaseException):
        return outcome
    if not isinstance(outcome, Exception):
        raise outcome
    logger.error('scenario %s raised outside its own failure handling', scenario_id, exc_info=outcome)
    print(f'[{_ts()}] [FAIL] {scenario_id}: {type(outcome).__name__}: {outcome} (no error.json — this raised outside the scenario runner)', file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Stability retry (qualification gate and final-suite annotation)
# ---------------------------------------------------------------------------

#: Why a scenario earned a second attempt. Typed, because the four cases are answered
#: differently and a boolean "should we retry" would collapse them.
StabilityReason = Literal['invalid_run', 'eval_failed', 'incomplete']

#: How stable the failure turned out to be, across the two attempts.
#:
#: ``flaky`` is the one that changes what a consumer may do: a scenario that fails once and
#: passes once is a worse MEASURING INSTRUMENT than one that fails twice, and it must not
#: trigger the expensive destructive step. In the qualification loop a redraw re-authors the
#: scenario and its LLM-named flow ids churn, which unscopes the evals written against it — so
#: a redraw fired on a coin flip costs a rebuild AND the eval set.
Stability = Literal['stable', 'flaky', 'unmeasured']

#: What the second sample said about ONE policy red — the grain the scenario verdict above
#: collapses. Per policy, because a scenario is a SET of reds and the set is what a reader acts
#: on: a first attempt red on ``[fidelity_judge, reply_length_budget]`` against a retry red on
#: ``[reply_length_budget]`` must not read ``stable`` while the fidelity red — the one a finding
#: is written from — has not reproduced.
#:
#: ``reproduced`` / ``not_reproduced`` are for the first attempt's reds, read against a
#: COMPARABLE retry (graded, and ended for no reason of its own); ``retry_only`` names a red the
#: retry added; ``unmeasured`` is a first-attempt red no comparable retry was taken for.
#:
#: ``retry_only`` is the one mark that can describe a CONTRADICTION rather than an addition: when
#: the first attempt GRADED the same policy and passed it, the run has two samples of that cell
#: and they disagree. The mark itself is unchanged by that reading — it still says "the retry red
#: a policy the first attempt did not" — but under a :data:`SampleKind` of ``judge_replay`` the
#: stability row NAMES the cell as contradicted (:func:`_contradicted_cells`). The retry still
#: decides nothing, the first attempt stays authoritative, and its transcript is never rewritten:
#: the row is the only record, and the backend reads it as an annotation the caller may overrule.
PolicyReproduction = Literal['reproduced', 'not_reproduced', 'retry_only', 'unmeasured']

#: What the second sample's own goal verdict (``scenario_verdict``) said beside the first's — an
#: annotation, never a verdict: ``retry_flipped`` is two conclusive verdicts that differ, ``agreed``
#: two that match, ``unmeasured`` any attempt without a conclusive one.
GoalReproduction = Literal['agreed', 'retry_flipped', 'unmeasured']

#: WHAT the second sample was, which decides what its marks are statements ABOUT. The marks and
#: the two functions that compute them are the same for both; only the input moves.
#:
#: ``agent_rerun`` re-ran the scenario end to end — a new thread, a new agent output — so its marks
#: compare two verdicts over two DIFFERENT outputs. That answers "is this failure a coin flip of
#: the agent's?" (the redraw question), and it cannot answer "did the judge flip?": over two
#: outputs both verdicts can be right (for example two batches sharing no generated sample, with a
#: copy judge self-consistent on each and opposite across them).
#:
#: ``judge_replay`` re-judged the FIRST attempt's own transcript — no agent, no user simulator — so
#: its marks compare two readings of ONE output. That answers "is this red judge noise?", and it is
#: the only kind under which two opposite verdicts on one cell are a CONTRADICTION. It can never
#: publish ``stable``: a judge agreeing with itself vouches for the judge, not for the failure.
SampleKind = Literal['judge_replay', 'agent_rerun']

#: Per kind, whether the second sample RE-RAN THE AGENT. The one fact every kind-keyed rule reads:
#: a contradiction is a judge contradiction only over an input that did NOT move, and a reproduction
#: licenses ``stable`` only over one that did.
_RESAMPLES_THE_AGENT: dict[SampleKind, bool] = {'agent_rerun': True, 'judge_replay': False}

#: WHICH second sample each phase takes when the operator names none (``--stability-sample``).
#: The two phases ask the two questions: a QUALIFICATION pass decides whether to redraw a
#: scenario, and a redraw fired on an agent coin flip costs a rebuild and the evals written against
#: it — so it resamples the agent. A FULL run already has its answer and asks whether each red is
#: the judge's noise — so it replays the judges over the first attempt's own transcript, which is
#: judge tokens only (no agent, no user simulator, no mock).
_PHASE_SAMPLE_KIND: dict[RunPhase, SampleKind] = {'qualification': 'agent_rerun', 'full': 'judge_replay'}

#: Fraction of COMPLETED scenarios that must carry an INFRASTRUCTURE signal before this run stops
#: spending retries.
#:
#: The shed exists so that a quota death does not double its own load: a retry storm
#: into a backend that is already failing makes the failure worse. It therefore reads the marks a
#: failing APPARATUS leaves on the rows — see :data:`InfrastructureSignal` — and never a judge's
#: red on its own. A strict suite where most scenarios fail some eval is the agent being
#: measured, not the apparatus failing, and a shed fed judge reds would cut off stability
#: measurement on exactly that suite while its sentence asserted an infrastructure cause nothing
#: had measured. The ratio protects retry spend only; it never moves a verdict.
_MASS_FAILURE_SHED_RATIO = 0.5

#: Below this many completed scenarios the ratio above is not yet meaningful (one flagged row out
#: of one is 100%), so the shed does not engage.
#:
#: It is a real floor, not a formality: a run smaller than it — an operator re-running one
#: scenario, the first two rows of any run — is under it, and for those the shed is not a check
#: that came back negative, it is a check that was never asked;
#: :meth:`_StabilityBudget.mass_failure_reading` says which.
_MASS_FAILURE_MIN_SAMPLE = 3

#: What the wholesale-failure shed had to say about this run, as a VALUE rather than a bare bool.
#: Below the sample floor the ratio is not ``False`` — it was never evaluated — and a line that
#: reports those two identically states a clean check on a run that never ran one.
MassFailureState = Literal['not_evaluated', 'engaged', 'clear']

#: The marks a completed row can carry that say the APPARATUS — not the agent — failed it. Each
#: is READ off a fact the row already records; none is inferred from a judge's verdict.
#:
#: * ``errored`` — the row produced no ``ThreadResult`` (its ``error.json`` is the record).
#: * ``incomplete`` — the per-scenario time budget cut it (``time_budget_exhausted``).
#: * ``empty_replies`` — EVERY turn emptied by the harness (``adapter_degraded``): the
#:   orchestrator's own whole-run classification. One emptied turn, or a blank reply, is the
#:   agent saying nothing — a verdict, not a signal (see :func:`_row_infrastructure_signals`).
#: * ``capacity`` — a reply that IS a provider's capacity error (a 429, an underscored
#:   ``rate_limit_error``, the OpenAI quota sentence): the agent returning its provider's
#:   refusal where its answer should be. Whole-reply, never a substring of free text.
#: * ``judges_errored`` — the finalize completed but the JUDGES failed the row: at least half of
#:   the cells it ASKED came back errored. The judge-fleet quota death, which grades nothing and
#:   wears a clean ``completed`` status.
#: * ``identical_failure_set`` — every graded row red on EVERY policy it was graded on, the same
#:   set on every row: the signature of the judges, or the replies they read, failing wholesale
#:   rather than of N scenarios each finding a defect. Cross-row by nature, so it is read over
#:   the ledger (:meth:`_StabilityBudget.infrastructure_rows`) rather than off one result — and
#:   on its own it sheds under its own name (``wholesale_reds``), because a small strict suite
#:   red everywhere is a reading about the AGENT as often as about the apparatus.
#: * ``identical_replies`` — every answering row said ONE thing, and the same one thing: the
#:   agent handed back a wall instead of an answer. Also cross-row, and it is the reading that
#:   costs no provider vocabulary (see :func:`_reads_as_a_status_line`).
InfrastructureSignal = Literal['errored', 'incomplete', 'empty_replies', 'capacity', 'judges_errored', 'identical_failure_set', 'identical_replies']

#: A provider's capacity error standing WHERE THE AGENT'S REPLY SHOULD BE — the agent caught its
#: provider's exception and returned ``str(exc)`` as its answer. Anchored at the START of the
#: stripped reply (leading punctuation and an exception-class prefix such as
#: ``anthropic.RateLimitError:`` allowed) and NEVER a substring over free reply text: an agent
#: whose domain is rate limiting says "our API gateway applies a rate limit per key" and "support
#: lines are overloaded" all day, and a substring match would read every such reply as the apparatus
#: failing. Only spellings a provider emits verbatim — the SDK error prefix, the underscored
#: error types, the 429 status line, Google's exhausted-resource line, and OpenAI's quota
#: sentence with or without its 429 prefix. The bare reason phrase is not here: "Too many
#: requests from one key is …" opens a sentence.
_CAPACITY_REPLY = re.compile(
    r'^\W*(?:[\w.]*error:\s*)?(?:'
    r'error code: (?:429|529)\b'
    r'|(?:openai\.|anthropic\.)?ratelimiterror\b'
    r'|rate_limit(?:_error)?\b'
    r'|overloaded_error\b'
    r'|resource_exhausted\b'
    r'|429 resource has been exhausted\b'
    r'|429 too many requests\b'
    r'|insufficient_quota\b'
    r'|(?:you )?exceeded your current quota\b'
    r')',
    re.IGNORECASE,
)

#: Why a wanted retry was NOT spent, typed onto the stability row as ``shed_reason`` so a reader
#: — and the plugin's renderer — never has to parse the sentence filed beside it. Three are the
#: shed proper: ``infrastructure`` (the per-row apparatus signals alone reached the ratio),
#: ``wholesale_reds`` (only the cross-row identical-failure reading did — the reds are the
#: agent's as often as the apparatus's, so it is never filed under the other name), and
#: ``retry_pool`` (the run-level pool ran dry). The other three are a retry that was never on
#: offer for this run (``disabled``: the phase or the flag; ``replay``: ``PLURAI_REPLAY`` makes a
#: second sample a replay of the first; ``no_budget``: the caller supplied no pool).
ShedReason = Literal['infrastructure', 'wholesale_reds', 'retry_pool', 'disabled', 'replay', 'no_budget']

#: Why an ``unmeasured`` row's retry SETTLED NOTHING, typed onto the row as ``unmeasured_reason``.
#: The OTHER SIDE of the fork :data:`ShedReason` names: that one says why a wanted retry was not
#: spent, this one why a retry that WAS spent decided nothing. A row carries exactly one of them.
#:
#: It exists because ``unmeasured`` is the RESIDUE bucket — one token over four distinct states,
#: three of which are reached with the retry already spent. Each state needs a typed field, so a
#: renderer never has to read the ABSENCE of ``shed_reason`` as a producer version. The sentence
#: in ``detail`` says the same thing in words; the typed fields are what a renderer branches on.
#:
#: ``no_decidable_reproduction``: the retry was graded and the comparison settled nothing — every
#: reproduction deterministic, or a retry that failed DIFFERENTLY.
#: ``retry_measured_nothing``: the retry ran but produced no comparison at all — it crashed, it
#: graded none of the first attempt's reds, or it ended for a reason of its own (a repeated
#: ``incomplete``). ``retry_failed``: the agent re-run itself FAILED before it was graded (a harness or
#: mock-runtime fault, a timeout) — the row's ``retry_failure`` carries the typed cause. ``first_attempt_invalid``:
#: the first attempt was never a measurement, so the retry that REPLACED it had nothing to compare against.
UnmeasuredReason = Literal['no_decidable_reproduction', 'retry_measured_nothing', 'retry_failed', 'first_attempt_invalid']

#: The cross-row members of :data:`InfrastructureSignal`, as the sets the ledger unions in.
_IDENTICAL_FAILURE_SET: frozenset[InfrastructureSignal] = frozenset({'identical_failure_set'})
_IDENTICAL_REPLIES: frozenset[InfrastructureSignal] = frozenset({'identical_replies'})

#: The cross-row signals derived from VERDICTS rather than from the apparatus. Subtracted before
#: :attr:`_StabilityBudget.mass_failure_kind` asks whether the apparatus alone reached the ratio,
#: because a suite red everywhere is the agent as often as the machinery — while a wall is a wall.
_VERDICT_DERIVED_SIGNALS: frozenset[InfrastructureSignal] = _IDENTICAL_FAILURE_SET

#: How long a reply may be and still read as a status line rather than as an answer. Generous:
#: a provider's error string with its JSON body attached runs to a couple of hundred characters,
#: and the identity requirement below is what does the actual discriminating.
_STATUS_LINE_MAX_CHARS = 400

#: A machine STATUS token, wherever it sits in the reply. Three shapes, none of them a provider's
#: name: an HTTP-family status number, an exception class (whatever word precedes ``Error`` /
#: ``Exception``), and an UPPER_SNAKE machine code. This is a GATE, not the signal — the signal is
#: cross-row reply identity, and the gate exists only to keep a canned agent refusal ("I can't
#: help with that", returned verbatim on every scenario) from being read as the apparatus when it
#: is a verdict about the agent.
_STATUS_TOKEN_RE = re.compile(
    r'\b[45]\d{2}\b'  # 429, 503, 529 … the status number itself
    r'|(?i:\b\w*(?:error|exception)\b)'  # ThrottlingException, RateLimitError, "error code"
    r'|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b'  # RESOURCE_EXHAUSTED, insufficient_quota's uppercase kin
)


def _is_provider_capacity_reply(reply: str) -> bool:
    """Is this agent reply a provider's capacity error, whole? See :data:`_CAPACITY_REPLY`."""
    return _CAPACITY_REPLY.match(reply.strip()) is not None


def _reads_as_a_status_line(reply: str) -> bool:
    """Could this reply be a machine's status rather than an answer to a user?

    The GATE on the cross-row identity reading, and deliberately weak: short, and carrying a
    status token somewhere in it. It is not a capacity detector and must not become one —
    :data:`_CAPACITY_REPLY` is that, and it stays the fast path for the per-row ``capacity``
    signal. What this pairs with is the property no provider spelling can hide: a capacity wall
    is THE SAME REPLY ON EVERY ROW, and an answer to a scenario is not byte-identical to the
    answer to a different one.

    That is why recognising a provider's quota error does not take one more regex alternation per
    provider. Bedrock's ``ThrottlingException``, Vertex's ``429 Quota exceeded``,
    Cohere's, Mistral's and Ollama's spellings, and any of them behind a wrapper's prefix, all
    satisfy this — as would the next provider's, unread by anyone here.
    """
    return len(reply) <= _STATUS_LINE_MAX_CHARS and _STATUS_TOKEN_RE.search(reply) is not None


def _distinct_replies(result: ThreadResult | None) -> frozenset[str]:
    """What this row's agent SAID, as a set of distinct non-empty replies.

    A set, because the question asked of it is identity, not order or count: a row that answered
    one thing all run has one member, and a row that held a conversation has several. A crashed
    row contributes nothing rather than an empty claim — it never reached an agent.
    """
    return frozenset(reply.strip() for reply in (result.chatbot_messages if result is not None else []) if reply.strip())


#: The judge-fleet reading's own sample floor: how many cells this row must have ASKED before
#: the share of them that errored means anything. One asked cell that errored is 100% of a
#: sample of one — the same noise :data:`_MASS_FAILURE_MIN_SAMPLE` exists to refuse one level up,
#: and the shape a row hits when the applicability guard skipped everything else.
_JUDGE_FLEET_MIN_ATTEMPTED = 2

#: The share of the ASKED cells that must have errored for the judges to have failed the row
#: rather than graded it. Half: at least as many errored as landed a verdict, counted over the
#: asked cells so the denominator says what it measures.
_JUDGE_FLEET_DEATH_RATIO = 0.5


def _judges_errored(result: ThreadResult) -> bool:
    """Did the JUDGES fail this row rather than grade it? The judge-fleet quota death leaves a
    finalize that ``completed`` with every cell ``status: 'error'``, which grades nothing and
    would otherwise read as a clean row.

    Measured against the cells that were ASKED — ``graded | errored`` — with a floor, not against
    GRADED alone: :func:`_graded_policies` subtracts the SKIPPED cells as well as the errored ones
    (rightly: its other four readers all want the cells that produced a measurement). A
    graded-only denominator would shrink when the applicability guard skipped most of a row's set
    until one broken judge was the whole fleet: one errored beside one graded would fire, and one
    errored beside NOTHING graded would fire hardest of all. Such a row would then count toward
    ``_MASS_FAILURE_SHED_RATIO``, and at half a run of three every failure in it would lose its
    stability retry under an ``infrastructure`` attribution nothing measured.

    The denominator is built HERE rather than in :func:`_graded_policies`, deliberately: that function is also
    ``_scenario_was_graded``, the per-policy stability record, the ``identical_failure_set``
    reading and ``judge_only``, and widening it to include errored cells would move four other
    readings.

    Below the floor the answer is ``False`` — no claim, the same three-state discipline
    :attr:`_StabilityBudget.mass_failure_state` applies to its own sample. The errored cells are
    still on the row's own record either way; what is refused is reading them as a fleet.
    """
    errored = _errored_policies(result)
    if not errored:
        return False
    attempted = errored | _graded_policies(result)
    if len(attempted) < _JUDGE_FLEET_MIN_ATTEMPTED:
        return False
    return len(errored) >= len(attempted) * _JUDGE_FLEET_DEATH_RATIO


def _row_infrastructure_signals(scenario_id: str, result: ThreadResult | None) -> frozenset[InfrastructureSignal]:
    """The infrastructure signals ONE row carries, read off facts it already records.

    Only the per-row members of :data:`InfrastructureSignal`; ``identical_failure_set`` needs
    the other rows and is read over the ledger. ``incomplete`` goes through :func:`_row_outcome`
    rather than re-comparing the reason, so the shed and the run summary cannot disagree about
    which rows the budget cut — and a QUARANTINED row, which that reading calls ``incomplete``,
    carries nothing here: the suite refused it before any apparatus touched it.

    What is deliberately NOT a signal: a turn the harness emptied for ``no_assistant_reply``, and
    a whitespace-only reply. Both are the agent producing nothing a user could read, which is the
    verdict ``response_not_empty`` exists to grade (``_protocol.adapter_error_is_scoreable``) —
    only the WHOLE-RUN case, every turn emptied, is the harness fault the orchestrator already
    types as ``adapter_degraded``.
    """
    if result is None:
        return frozenset() if _unmeasured_by_design(scenario_id, result) else frozenset({'errored'})
    signals: set[InfrastructureSignal] = set()
    if _row_outcome(result) == 'incomplete':
        signals.add('incomplete')
    if result.ended_early == ADAPTER_DEGRADED_REASON:
        signals.add('empty_replies')
    if any(_is_provider_capacity_reply(reply) for reply in result.chatbot_messages):
        signals.add('capacity')
    if _judges_errored(result):
        signals.add('judges_errored')
    return frozenset(signals)


@dataclass
class _StabilityBudget:
    """The run-level pool of stability retries, and every reason one was NOT spent.

    A POOL rather than one-per-scenario: at most half the suite may be retried, so a run where
    everything fails cannot silently cost twice what the operator asked for. Exhaustion is not a
    quiet no-op — every shed is recorded against the scenario that wanted it, because a
    ``stability`` field that is simply absent reads as "this scenario was never in doubt".

    The pool also sheds under MASS FAILURE, and that check reads the apparatus, not the verdicts:
    a row counts toward it only for an :data:`InfrastructureSignal` it actually carries. A suite
    of judge reds — however many — keeps measuring stability, because measuring it is what the
    retries are for on exactly that suite.

    An INVALID run (the simulated user repeated itself) is exempt from the pool: that
    attempt was never a measurement, so re-running it is not a second sample, it is the first
    one. Charging it would let a broken simulator eat the budget real failures need.
    """

    remaining: int
    spent: int = 0
    #: Per completed scenario, the per-row infrastructure signals it carried — empty for a clean
    #: row and for a judge-red-only row alike. Its length is the shed's denominator.
    signals: dict[str, frozenset[InfrastructureSignal]] = dataclasses_field(default_factory=dict)
    #: Per completed scenario, ``(graded policies, violated policies)`` — the input to the one
    #: CROSS-ROW signal, and to the reading's count of rows that failed on judge verdicts alone.
    #: A row whose finalize named no violated set at all records an empty one here: for these
    #: two readings "no reds on record" and "no reds" agree.
    verdicts: dict[str, tuple[frozenset[str], frozenset[str]]] = dataclasses_field(default_factory=dict)
    #: Per completed scenario, the DISTINCT things its agent said — the input to the other
    #: cross-row signal. Empty for a row that never answered.
    replies: dict[str, frozenset[str]] = dataclasses_field(default_factory=dict)
    shed: dict[str, str] = dataclasses_field(default_factory=dict)
    #: The typed twin of :attr:`shed` — written together with the sentence by :meth:`claim`.
    shed_reasons: dict[str, ShedReason] = dataclasses_field(default_factory=dict)

    @property
    def completed(self) -> int:
        return len(self.signals)

    def infrastructure_rows(self) -> dict[str, frozenset[InfrastructureSignal]]:
        """Per completed scenario, EVERY infrastructure signal it carries, the cross-row ones included.

        ``identical_failure_set`` is decided here because it is a fact about the ledger: every
        graded row (at least the sample floor of them) red on every policy it was graded on, with
        the same policy set — of more than one policy — on every row. One row red on everything
        is an agent that failed badly; every row red on everything, identically, is the apparatus.

        ``identical_replies`` is the same kind of fact read one channel over, and it is how a
        capacity wall is recognised without knowing which provider built it: every row that
        answered said ONE thing, and every row said the SAME one thing. A chat answer is not
        byte-identical across scenarios — the personas differ, the world differs, the questions
        differ — so identity across a run is a statement about the machinery, not the agent. The
        status-line gate (:func:`_reads_as_a_status_line`) is what keeps a canned refusal, which
        IS a verdict about the agent, on the agent's side of that line.
        """
        rows = dict(self.signals)
        # Graded rows only: a dead-judge row (nothing graded) carries its own signal and must not
        # break — or make — the identity of the rows that were graded.
        graded_rows = {sid: pair for sid, pair in self.verdicts.items() if pair[0]}
        if len(graded_rows) >= _MASS_FAILURE_MIN_SAMPLE and len(set(graded_rows.values())) == 1:
            graded, violated = next(iter(graded_rows.values()))
            # Two cells at least: "red on every policy" with one policy is "red on the policy",
            # and a one-policy set has no second cell to tell wholesale from strict.
            if violated == graded and len(graded) > 1:
                for sid in graded_rows:
                    rows[sid] = rows[sid] | _IDENTICAL_FAILURE_SET
        # Answering rows only, for the same reason: a row that produced no reply at all carries
        # its own signal and says nothing about what the others were told.
        answering_rows = {sid: replies for sid, replies in self.replies.items() if replies}
        if len(answering_rows) >= _MASS_FAILURE_MIN_SAMPLE and len(set(answering_rows.values())) == 1:
            said = next(iter(answering_rows.values()))
            if len(said) == 1 and _reads_as_a_status_line(next(iter(said))):
                for sid in answering_rows:
                    rows[sid] = rows[sid] | _IDENTICAL_REPLIES
        return rows

    def signal_counts(self) -> dict[InfrastructureSignal, int]:
        """How many completed rows carry each signal — the MEASUREMENT every shed sentence states."""
        counts: dict[InfrastructureSignal, int] = {}
        for signals in self.infrastructure_rows().values():
            for signal in signals:
                counts[signal] = counts.get(signal, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def mass_failure_state(self) -> MassFailureState:
        """Whether the wholesale-failure shed applies, did NOT apply, or could not be asked.

        Three answers, because the floor exists to make the third one real: one flagged row out
        of one is 100%, so under ``_MASS_FAILURE_MIN_SAMPLE`` completed scenarios the ratio is
        noise rather than a negative result.
        """
        if self.completed < _MASS_FAILURE_MIN_SAMPLE:
            return 'not_evaluated'
        flagged = sum(1 for signals in self.infrastructure_rows().values() if signals)
        return 'engaged' if flagged >= self.completed * _MASS_FAILURE_SHED_RATIO else 'clear'

    @property
    def mass_failure(self) -> bool:
        """The shed decision. DERIVED from :attr:`mass_failure_state`, so the branch taken and
        the state rendered beside it cannot disagree — the failure mode of every floor that is
        one expression in the code and another in the report."""
        return self.mass_failure_state == 'engaged'

    @property
    def mass_failure_kind(self) -> Literal['infrastructure', 'wholesale_reds']:
        """WHICH reading engaged the shed: the apparatus signals on their own, or only the
        cross-row identical-failure set. It is what the refusal is filed under — a small strict
        suite red everywhere is the agent as often as the apparatus, and filing it as
        ``infrastructure`` would assert the latter.

        The split is by what the signal is DERIVED from, not by whether it is per-row:
        :data:`_VERDICT_DERIVED_SIGNALS` is subtracted, so ``identical_replies`` — cross-row, but
        read off the reply channel rather than off any judge's verdict — counts as apparatus.
        """
        apparatus = sum(1 for signals in self.infrastructure_rows().values() if signals - _VERDICT_DERIVED_SIGNALS)
        return 'infrastructure' if apparatus >= self.completed * _MASS_FAILURE_SHED_RATIO else 'wholesale_reds'

    def mass_failure_reading(self) -> str:
        """The check's state in words, for the sentence filed against a scenario.

        States what was MEASURED — the flagged rows, by signal, and the rows that failed on judge
        verdicts alone — and never a cause: "one infrastructure event" is a claim nothing here
        has checked. Rendered on every shed, including — especially — when the floor kept the
        check from running: "the pool is exhausted" alone reads as though the wholesale-failure
        check had been asked and answered no, which is false on a run of two scenarios.
        """
        state = self.mass_failure_state
        if state == 'not_evaluated':
            return f'mass-failure check below sample floor, n={self.completed}<{_MASS_FAILURE_MIN_SAMPLE} — not evaluated'
        rows = self.infrastructure_rows()
        flagged = sum(1 for signals in rows.values() if signals)
        # A row failed on judge verdicts alone when a policy it was GRADED on came back red and
        # nothing about the apparatus is on the row — a red on a cell whose judge errored is
        # not a verdict, and a dead-judge row is not a judge-only failure.
        judge_only = sum(1 for sid, (graded, violated) in self.verdicts.items() if violated & graded and not rows[sid])
        measured = f'{flagged} of {self.completed} completed scenarios carry infrastructure signals'
        if counts := self.signal_counts():
            measured += ' (' + ', '.join(f'{signal} x{n}' for signal, n in counts.items()) + ')'
        if judge_only:
            measured += f'; {judge_only} failed on judge verdicts alone'
        return f'mass failure: {measured}' if state == 'engaged' else f'mass-failure check clear: {measured}'

    def observe(self, scenario_id: str, result: ThreadResult | None) -> None:
        """Record one completed scenario: the signals its row carries, and what it was graded on.

        Takes the RESULT, not a verdict: a verdict ("did this scenario earn a retry") would count
        every judge red as evidence of a failing apparatus. A crashed row
        (``None``) is the clearest infrastructure signal there is and is counted here too — the
        stability retry itself still never re-drives a crash (see :func:`_stability_reason`).

        A QUARANTINED row is not in the census at all: the suite refused it before it ran, so it
        is neither a completed scenario nor evidence about the apparatus — counted, it would
        read as a crash and shed retries on a run whose backend is fine.
        """
        if _unmeasured_by_design(scenario_id, result):
            return
        self.signals[scenario_id] = _row_infrastructure_signals(scenario_id, result)
        self.verdicts[scenario_id] = (_graded_policies(result), _violated_policies(result) or frozenset())
        self.replies[scenario_id] = _distinct_replies(result)

    def note(self, scenario_id: str, sentence: str) -> None:
        """File one per-scenario sentence in :attr:`shed`, KEEPING any already filed.

        Additive rather than an assignment because two subsystems write here — this pool's own
        refusals and the artifact hygiene that runs before the first attempt
        (:func:`_clear_stale_retry_dir`) — and a plain assignment by whichever ran second would erase
        the other. The erased sentence would be unrecoverable: this dict is the only place it exists.
        """
        prior = self.shed.get(scenario_id)
        self.shed[scenario_id] = f'{prior}; also: {sentence}' if prior else sentence

    def claim(self, scenario_id: str, reason: StabilityReason) -> bool:
        """May ``scenario_id`` spend a retry for ``reason``? Records the refusal — sentence AND
        typed reason — when not."""
        if reason == 'invalid_run':
            return True  # free: the first attempt was never a measurement
        if self.mass_failure:
            kind = self.mass_failure_kind
            why = (
                'a retry into a failing apparatus doubles its load'
                if kind == 'infrastructure'
                else (
                    'every graded scenario is red on every policy, identically, so a second sample of one re-measures the wholesale '
                    'condition, not that scenario'
                )
            )
            self.note(scenario_id, f'shed: {self.mass_failure_reading()} — {why}, so this failure was not re-sampled')
            self.shed_reasons[scenario_id] = kind
            return False
        if self.remaining <= 0:
            # The mass-failure state travels WITH this refusal. Without it the line is a clean
            # one about a check that, on any run below the sample floor, was never evaluated.
            self.note(scenario_id, f'shed: the run-level stability-retry pool is exhausted ({self.mass_failure_reading()})')
            self.shed_reasons[scenario_id] = 'retry_pool'
            return False
        self.remaining -= 1
        self.spent += 1
        return True


def _stability_budget_for(scenario_ids: Sequence[str], run_phase: RunPhase = 'full', *, sample_kind: SampleKind = 'agent_rerun') -> _StabilityBudget:
    """The run-level retry pool for a run of ``scenario_ids``, sized by N, the phase AND the kind.

    ``full`` with an ``agent_rerun``: half the suite, rounded up, never zero. Half because an agent
    re-run costs a whole scenario's spend and the operator asked for one run, not two; never zero
    because a single-scenario re-run is exactly the case an operator debugging one failure wants.

    A ``judge_replay`` (:data:`SampleKind`): N, in either phase. It spends judge tokens only — no
    agent, no user simulator, no mock — so the reason for halving does not apply, and halving
    anyway would leave half a full run's reds shed as ``retry_pool``: unmeasured by arithmetic.

    ``qualification``: N. A qualification set is THREE scenarios, so half-rounded-up is 2 — and
    when all three fail one of them is ALWAYS shed as ``retry_pool``, by construction, before
    anything about that scenario is known. Every critic pass then reads ``stability: unmeasured``
    as a caveat on the very row it is auditing, on a set small enough that the pool was never the
    scarce thing. A pool >= N cannot manufacture that row; on a 3-scenario set the worst case is
    3 extra scenario runs, which is the budget the phase exists to spend.

    An INVALID run is exempt from the pool entirely — see :meth:`_StabilityBudget.claim`.
    """
    total = len(scenario_ids)
    whole_suite = run_phase == 'qualification' or not _RESAMPLES_THE_AGENT[sample_kind]
    return _StabilityBudget(remaining=max(1, total if whole_suite else (total + 1) // 2))


def _replay_mode() -> str:
    """The backend replay mode in force for THIS process, or ``''``.

    A local end-to-end runs the SDK and the backend in one environment, so this is visible here
    — and it is the one condition under which a retry is worthless: replay serves the recorded
    fixture for the same scenario and turn, so the second attempt reproduces the first exactly
    and "stable" would be an artifact of the recording rather than a fact about the agent.
    """
    return (os.environ.get('PLURAI_REPLAY') or '').strip().lower()


def _violated_policies(result: ThreadResult | None) -> frozenset[str] | None:
    """WHICH policies this attempt violated, or ``None`` when the finalize named no set at all.

    ``_stability_reason`` collapses any violation to the coarse ``eval_failed``; the redraw
    decision and the per-policy record need the names. The two absences are kept apart on
    purpose: ``[]`` is the server stating "no red", ``None`` is "no statement" — collapsed, a
    retry whose payload carried no list would read as a clean retry, and every first-attempt red
    would become ``not_reproduced`` and the scenario ``flaky`` on evidence of nothing.
    """
    per = (result.finalize.policy_evaluation_results or {}) if result is not None else {}
    violated = per.get('violated_policies')
    return frozenset(str(v) for v in violated) if isinstance(violated, list) else None


def _stability_reason(result: ThreadResult | None) -> StabilityReason | None:
    """Why this scenario earns a second attempt, or ``None``.

    Order is precedence, most specific first. An INVALID run outranks everything: it is not a
    failing measurement, it is the absence of one, so nothing else about it can be read.

    A scenario that never produced a ``ThreadResult`` is deliberately absent from this list. A
    crash is the RE-DRIVE's business (``orchestrator/redrive``), which already spends one
    attempt on exactly the failures a fresh thread can survive; a second mechanism retrying the
    same class would spend two attempts on one transient and none on the flakiness this exists
    to measure.
    """
    if result is None:
        return None
    per = result.finalize.policy_evaluation_results or {}
    user_sim = per.get('user_sim_integrity')
    if isinstance(user_sim, dict) and user_sim.get('run_valid') is False:
        return 'invalid_run'
    if _row_outcome(result) == 'incomplete':
        return 'incomplete'
    if not _scenario_was_graded(result):
        # Nothing was measured, so there is no failure to characterise. Re-running would not
        # produce a verdict either — the eval set, not the sample, is what is missing.
        return None
    # Read through the same accessor the verdict compares names with, so the two can never
    # disagree about what counts as a violation — this frame only collapses them to one reason.
    return 'eval_failed' if _violated_policies(result) else None


def _decided(cell: Mapping[str, Any]) -> bool:
    """Whether this cell is a VERDICT: the backend's own rule, an allowlist of one status.

    Only ``success`` is conclusive; an ``inconclusive`` cell of any reason (N/A, deferred, a judge
    that abstained, a contested or weakly agreed one) enters no rollup on the producer side, so it
    is not a sample of a red here either.
    """
    return cell.get('status') == 'success'


def _policy_reproduction(first: ThreadResult | None, second: ThreadResult | None, reason: StabilityReason) -> dict[str, PolicyReproduction]:
    """Per policy red on either attempt, what the second sample said about it.

    Keyed by the first attempt's reds plus any the retry added, sorted. Comparability is decided
    at TWO grains, and both have to hold before a red can read ``reproduced`` or
    ``not_reproduced``:

    * the ATTEMPTS. ``reason`` is why the first attempt earned a retry; only ``eval_failed``
      makes the first attempt a sample of its reds (a first attempt cut by the clock, or
      invalid, has reds about the turns it reached, or none worth naming — an INVALID attempt
      writes no map at all, whichever path records it). The retry likewise: one that crashed,
      was not graded, or ended for a reason of its own measures nothing, and no ``retry_only``
      entry is written for it. ``second=None`` is therefore also how a scenario REFUSED a retry
      records which reds it never re-sampled.
    * the POLICY. A red is measured only where the retry actually GRADED that policy
      (:func:`_graded_policies`): a judge that errored on the retry, or a policy its
      applicability guard skipped, says nothing about the red, and reading its absence as
      ``not_reproduced`` would make the scenario ``flaky`` on a verdict nobody gave. A retry whose
      finalize named no violated set at all (:func:`_violated_policies` → ``None``) is the same
      absence one level up.
    """
    if reason == 'invalid_run':
        return {}
    first_reds = _violated_policies(first) or frozenset()
    if reason != 'eval_failed':
        second = None
    second_reds = _violated_policies(second) if second is not None and _stability_reason(second) in (None, 'eval_failed') else None
    if second_reds is None:
        return dict.fromkeys(sorted(first_reds), 'unmeasured')
    # SCOPED TO THIS FUNCTION ON PURPOSE. `_graded_policies` itself also backs `_scenario_was_graded`
    # (retry eligibility, below) and the mass-failure shed; subtracting N/A there would silently stop
    # retrying premise-gated scenarios. Here the question is
    # narrower and has one answer: what did the retry SAY about this red. A cell that is not a verdict said nothing.
    cells = _policy_cells(second)
    graded = frozenset(name for name in _graded_policies(second) if _decided(cells.get(name, {})))
    reproduction: dict[str, PolicyReproduction] = {
        name: ('reproduced' if name in second_reds else 'not_reproduced') if name in graded else 'unmeasured' for name in sorted(first_reds)
    }
    reproduction.update(dict.fromkeys(sorted(second_reds - first_reds), 'retry_only'))
    return reproduction


def _contradicted_cells(first: ThreadResult | None, reproduction: Mapping[str, PolicyReproduction]) -> frozenset[str]:
    """The policies BOTH attempts graded and graded OPPOSITE ways — a pure read, no I/O.

    Rides :func:`_policy_reproduction`'s own map rather than comparing the two attempts again.
    That map has already decided what is comparable at both grains — a retry cut by the clock or
    ended for a reason of its own writes no ``retry_only`` entry at all — so a dialog that simply
    went further on the retry and reached a surface the first never did cannot land here.

    ``retry_only`` alone is not enough, because it fires on ``second_reds - first_reds``: the first
    attempt may have SAID NOTHING about that policy (skipped, errored, not applicable, or a cell
    the server named as errored), and there is no verdict to contradict where none was given. So the
    first attempt must have GRADED it, on the house predicate, and its cell must be a decided one
    (``status == 'success'``).

    KEYED ON THE CONTRADICTION AND NOTHING ELSE. Not on agreement, not on confidence, not on
    whether a judge or a deterministic expression produced the cell, and not on a policy name: a
    cell can carry no ``agreement`` key at all and an empty ``usage_metadata``, and is exactly as
    contradicted as the multi-sample ones beside it.

    A pure comparison, so it answers the same way for either :data:`SampleKind` — and it is only
    ASKED under ``judge_replay``, at the one call site. Over an ``agent_rerun`` the two verdicts
    graded two different outputs and may both be right, so opposite answers there are not a
    contradiction of anything.
    """
    cells = _policy_cells(first)
    graded = _graded_policies(first)
    return frozenset(name for name, mark in reproduction.items() if mark == 'retry_only' and name in graded and _decided(cells[name]))


#: The evidence label for a red no sampling was done on — a `code_eval`. On a ``judge_replay`` its
#: retry re-runs the SAME simpleeval expression over the SAME saved shape (an ``agent_rerun`` grades
#: a NEW transcript, so its mark is a real measurement), so `✔ reproduced` is a tautology, not
#: a measurement, and would bless a deterministic FALSE red as
#: "stable" — the verdict that licenses a redraw, i.e. a re-authored scenario and every eval
#: written against it unscoped, bought with no evidence at all.
DETERMINISTIC_EVIDENCE = 'deterministic'


#: The evidence label for a judge that RAN but took one sample — `policy_judge_samples: 1`, the
#: documented cheap setting. It reports no agreement fraction, so the pair below would render
#: `agreement — → —`, which is byte-identical to a policy neither attempt graded. Two different
#: runs must not print the same token: this one says a verdict WAS reached, by an LLM, from a
#: single draw — so the retry beside it IS an independent sample and its mark means something.
JUDGED_UNSAMPLED_EVIDENCE = 'judged, unsampled'


def _cell_agreement(cells: Mapping[str, dict[str, Any]], name: str) -> float | None:
    """One policy cell's self-consistency agreement on this attempt, or ``None``.

    Present only where K>1 sampling ran (the backend's judge-sample aggregation), which is every
    `llm_as_judge` under the default `policy_judge_samples: 3` and never a `code_eval` — the backend
    forces K=1 for its deterministic agents because K samples of a fixed expression cost K times
    the work to reproduce one answer.

    Takes the CELLS, not the attempt: :func:`_policy_cells` is a re-derivation off the payload and
    this is called twice per policy per attempt, so reading it here would make the caller's one
    loop rebuild the same map 4N times.
    """
    value = cells.get(name, {}).get('agreement')
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cell_was_judged(cells: Mapping[str, dict[str, Any]], name: str) -> bool:
    """Did an LLM produce this cell? Read off the cell's own token usage.

    The corroborating half of the discriminator below. Agreement alone would call a judge running
    at `policy_judge_samples: 1` — the documented cheap setting — `deterministic`, which is
    a false claim about how its verdict was reached: that judge's retry IS an independent sample.
    A cell with no agreement AND no judge tokens spent is the deterministic one.
    """
    return bool(cells.get(name, {}).get('usage_metadata'))


def _policy_evidence(first: ThreadResult | None, second: ThreadResult | None, names: Iterable[str]) -> dict[str, str]:
    """Per named red, WHAT the reproduction mark beside it is evidence of.

    A mark on its own is not a measurement, and the two kinds of policy need opposite readings:

    * a `code_eval` is :data:`DETERMINISTIC_EVIDENCE`. On a ``judge_replay`` it re-runs one
      expression over one saved shape and cannot disagree, so the renderer prints that INSTEAD of
      a reproduction mark; on an ``agent_rerun`` it graded a new transcript, and the mark stays.
    * an `llm_as_judge` samples K times per attempt and reports the majority fraction, so the
      honest evidence is the PAIR: `0.67 → 1.00` says the retry was more certain than the first
      attempt, and `✔ reproduced` on its own is P(red)² — a judge that catches a real defect 60%
      of the time reads "not reproduced" 40% of the time it is right.

    An attempt that never graded the policy contributes ``—``: there is a number missing, and
    printing one side of a pair as though it were both is how an unmeasured half reads as an
    agreement of zero. A judge that ran at K=1 has no number on EITHER side, which the pair would
    render `— → —` — indistinguishable from a policy nobody graded — so it gets its own token
    (:data:`JUDGED_UNSAMPLED_EVIDENCE`); the cell's own token usage is what tells that apart from
    the deterministic case.

    Both attempts' cells are read ONCE, here, and handed down. The two readers below are called
    per policy per attempt, and each re-derivation walks the whole finalize payload.
    """
    first_cells, second_cells = _policy_cells(first), _policy_cells(second)
    evidence: dict[str, str] = {}
    for name in names:
        before, after = _cell_agreement(first_cells, name), _cell_agreement(second_cells, name)
        if before is None and after is None:
            judged = _cell_was_judged(first_cells, name) or _cell_was_judged(second_cells, name)
            evidence[name] = JUDGED_UNSAMPLED_EVIDENCE if judged else DETERMINISTIC_EVIDENCE
            continue
        evidence[name] = f'agreement {_agreement_token(before)} → {_agreement_token(after)}'
    return evidence


def _agreement_token(value: float | None) -> str:
    """One side of the pair. ``—`` for an attempt that did not grade the policy — an ABSENT
    measurement, which `0.00` would report as unanimous disagreement."""
    return '—' if value is None else f'{value:.2f}'


#: The marks that COMPARE the two attempts. ``unmeasured`` is the fourth value and it compares
#: nothing — a row carrying only those (a refused retry, or one that ended for a reason of its
#: own) has no comparison for a validity claim to qualify.
_COMPARING_MARKS: frozenset[str] = frozenset({'reproduced', 'not_reproduced', 'retry_only'})


def _unstamped_clause(first_revision: str | None, second_revision: str | None) -> str:
    """Which side of the pair carries no eval revision, as the CLAUSE the reason opens with.

    The whole clause rather than the subject: the both-unstamped case needs "neither attempt
    carries AN eval revision", and a shared "carries NO eval revision" tail would turn that into a
    double negative reading as its own opposite — both attempts carrying one, which is the state
    in which no disproof is written at all.
    """
    if not (first_revision or second_revision):
        return 'neither attempt carries an eval revision'
    return f'{"the first attempt" if not first_revision else "the retry"} carries NO eval revision'


def _marks_comparability(first: ThreadResult | None, second: ThreadResult | None, *, marks: Mapping[str, PolicyReproduction] | None) -> dict[str, Any]:
    """WHICH eval revision graded each attempt, and — where they disagree — that the marks beside
    them compare verdicts two different eval sets produced.

    A ``policy_reproduction`` mark is a comparison of two attempts, so it means something only if
    the SAME judges graded both. ``eval_revision`` — the backend's content digest over the set's
    per-policy fingerprints — is what says so, so it is recorded per ATTEMPT: the
    run summary's ``eval_revisions`` map is the AUTHORITATIVE attempt's only. Without the second
    sample's, a reader would have to decide comparability from file mtimes.

    The revisions are stamped whenever an attempt carried one, and OMITTED (never defaulted) when
    it did not — a backend that stamps none and a finalize that scored nothing both carry none,
    and a blank one would read as a revision.

    ONLY THE NEGATIVE IS WRITTEN. ``marks_valid: False`` (with its reason) says this runner
    established that the marks do not compare one measurement; its ABSENCE says it found no such
    disproof, and the two stamps beside it are what a reader compares for itself. A positive
    ``marks_valid: True`` would be a claim a LATER event falsifies — a rescore of one attempt
    after the run, which this writer cannot see — and it would then sit in the published row
    beside the consumer's own ``marks_withdrawn``, asserting both at once.

    Claimed only where there IS a comparison to qualify: a row whose marks are every one
    ``unmeasured`` compared nothing, whatever the revisions say.
    """
    first_revision, second_revision = _eval_revision_of(first), _eval_revision_of(second)
    row: dict[str, Any] = {
        **({'first_eval_revision': first_revision} if first_revision else {}),
        **({'second_eval_revision': second_revision} if second_revision else {}),
    }
    if not any(mark in _COMPARING_MARKS for mark in (marks or {}).values()):
        return row
    if not (first_revision and second_revision):
        row['marks_valid'] = False
        row['marks_invalid_reason'] = (
            f'{_unstamped_clause(first_revision, second_revision)} on the finalize that scored it, so whether the two '
            'samples were graded by the same judges cannot be established — and an unverifiable comparison is not a '
            'valid one'
        )
    elif first_revision != second_revision:
        row['marks_valid'] = False
        row['marks_invalid_reason'] = (
            f'the first attempt was graded at eval revision {first_revision} and the retry at {second_revision}, so '
            'every per-policy mark on this row compares verdicts two different eval sets produced'
        )
    return row


def _stability_verdict(
    second: ThreadResult | None,
    reason: StabilityReason,
    reproduction: Mapping[str, PolicyReproduction],
    *,
    evidence: Mapping[str, str],
    sample_kind: SampleKind = 'agent_rerun',
) -> Stability:
    """What the second sample says about the first attempt's failure.

    THREE outcomes, and the third is the one that costs money if it is folded into either
    neighbour. ``stable`` LICENSES A REDRAW, so it is claimed only when the retry reproduced
    THE SAME failure; ``flaky`` only when the retry came back clean AND graded — that pair is
    the whole evidence that the scenario is a wobbly instrument rather than a defect detector.

    For ``eval_failed`` the verdict is DERIVED from the per-policy record (``reproduction``, see
    :func:`_policy_reproduction`, and ``evidence``, see :func:`_policy_evidence`) rather than
    computed beside it, so the row's scenario verdict and its per-policy marks cannot disagree: a
    red that reproduced on a SECOND SAMPLE is ``stable``; every red gone — measured gone, on a
    policy the retry graded — and none added is ``flaky``; anything else — a red the retry could
    not measure, a retry that failed DIFFERENTLY (attempt 1 violating ``no_refund_without_auth``
    and attempt 2 ``must_greet_by_name`` is two things going wrong once each, and reading that as
    ``stable`` redraws a scenario, unscoping every eval written against it, on evidence of
    nothing), or a retry that measured nothing — is ``unmeasured``.

    A :data:`DETERMINISTIC_EVIDENCE` REPRODUCTION IS NOT A SECOND SAMPLE. A ``code_eval`` re-runs
    one expression, so a check that is wrong is wrong on both attempts and its ``reproduced`` cannot
    tell a false red from a true one. It cannot carry ``stable`` on its own: a row whose only
    reproductions are deterministic took no sample that could have disagreed, and is ``unmeasured``.
    Nothing is dropped from the row — its mark and evidence stay in ``policy_reproduction`` / ``policy_evidence`` —
    and a judged reproduction beside it still makes the row ``stable``.

    A ``judge_replay`` REPRODUCTION IS NOT A SECOND SAMPLE OF THE FAILURE EITHER — the same rule, one
    term wider (:data:`SampleKind`). It re-judged the first attempt's own transcript, so a judge
    that agrees with itself corroborates the INSTRUMENT, not the agent's failure, and ``stable``
    over it would license a redraw on evidence of nothing. A replay may still come back ``flaky``
    (the judge withdrew every red on the same input — the instrument wobbles) or ``unmeasured``;
    only an ``agent_rerun`` can reproduce a failure.

    ``sample_kind`` is keyword-only and defaults to ``agent_rerun`` for ONE caller outside this
    module: the plugin's rescore recompute (``build_run_json.recomputed_marks``) derives this
    verdict over a ``stability_retry/`` pair — two agent attempts by construction — and an
    installed plugin calls without it. Every call site in this package passes it explicitly.

    ``incomplete`` twice is never ``stable`` either. A scenario cut by the TIME BUDGET twice is
    a budget fact, and the remedy for it is a bigger budget (or ``--rerun-on-turn-cap``), not
    re-authoring the scenario.
    """
    if reason == 'eval_failed':
        # Indexed, never `.get`: the caller builds ``evidence`` over these same names, and an entry
        # defaulted to "not deterministic" would hand `stable` to exactly the red this rule withholds it from.
        if _RESAMPLES_THE_AGENT[sample_kind] and any(mark == 'reproduced' and evidence[name] != DETERMINISTIC_EVIDENCE for name, mark in reproduction.items()):
            return 'stable'
        return 'flaky' if set(reproduction.values()) == {'not_reproduced'} else 'unmeasured'
    if second is None or _stability_reason(second) is not None:
        # A repeated ``incomplete`` (or a repeated ``invalid_run``) reproduced, but not a SCENARIO
        # defect — nothing here licenses a redraw; and a retry that failed differently is
        # two measurements of two things.
        return 'unmeasured'
    return 'flaky' if _scenario_was_graded(second) else 'unmeasured'


def _unmeasured_reason(
    second: ThreadResult | None,
    reason: StabilityReason,
    reproduction: Mapping[str, PolicyReproduction],
    *,
    evidence: Mapping[str, str],
    sample_kind: SampleKind = 'agent_rerun',
) -> UnmeasuredReason | None:
    """WHICH branch of :func:`_stability_verdict` produced ``unmeasured``, as a typed value.

    A sibling rather than a second return value from the verdict itself: the verdict is the
    contract three consumers read. This asks the verdict first, so the two cannot disagree, and
    returns ``None`` for every row the verdict does not send here.

    The split inside ``eval_failed`` is the same distinction :data:`_COMPARING_MARKS` already
    draws: a mark that compares two attempts exists only when the retry GRADED that red, so a map
    holding one means the comparison ran and settled nothing, and a map holding none means the
    retry measured nothing to compare. The ``invalid_run`` branch is not reachable from here — it
    never consults the verdict (see :func:`_drive_scenario_with_stability`) and names itself.

    ``sample_kind`` is handed straight to the verdict, so a ``judge_replay`` whose judge agreed —
    every mark ``reproduced``, the verdict withheld from ``stable`` — reads the existing
    ``no_decidable_reproduction``: a comparison ran and settled nothing about the failure.
    """
    if _stability_verdict(second, reason, reproduction, evidence=evidence, sample_kind=sample_kind) != 'unmeasured':
        return None
    if reason == 'eval_failed' and any(mark in _COMPARING_MARKS for mark in reproduction.values()):
        return 'no_decidable_reproduction'
    return 'retry_measured_nothing'


def _failed_retry_reason(reason: UnmeasuredReason | None, retry_failure: Mapping[str, Any] | None) -> UnmeasuredReason | None:
    """``retry_failed`` where the residue ``retry_measured_nothing`` was reached by an agent re-run that FAILED."""
    return 'retry_failed' if reason == 'retry_measured_nothing' and retry_failure is not None else reason


@dataclass(frozen=True, slots=True)
class _RetryRefusal:
    """Why a wanted retry was not spent: the typed reason, its sentence, and — for the two
    mass-failure sheds — the per-signal counts the decision was made on. All three reach the
    stability row, so the row states what was measured rather than a cause."""

    reason: ShedReason
    detail: str
    signals: dict[InfrastructureSignal, int] = dataclasses_field(default_factory=dict)


def _retry_refusal(cfg: _RunConfig, budget: _StabilityBudget | None, scenario_id: str, reason: StabilityReason) -> _RetryRefusal | None:
    """Why this scenario does NOT get its second sample, or ``None`` when it may have one.

    Every branch returns the refusal FILED AGAINST THE SCENARIO, because that record is the
    whole account of why a retry was not spent — a stability row that is simply absent reads as
    "this scenario was never in doubt", and each of these was.

    Ordered, and the pool is CLAIMED LAST: a retry refused for any earlier reason must cost the
    pool nothing.
    """
    if budget is None:
        # No pool was supplied (a caller that predates it, or a test double).
        return _RetryRefusal('no_budget', 'not retried: this run was given no stability-retry budget')
    if reason != 'invalid_run' and not cfg.stability_retry_enabled:
        return _RetryRefusal('disabled', 'not retried: stability retries are disabled for this run (--no-stability-retry)')
    if reason != 'invalid_run' and cfg.single_hop(scenario_id):
        # A single-hop replay takes its samples as --trials: a retry would add an unequal one.
        return _RetryRefusal('disabled', 'not retried: a single-hop regression replay takes its samples as --trials')
    if (mode := _replay_mode()) in ('replay', 'record'):
        # A replayed retry resolves the SAME recorded fixture for the same scenario and turn, so
        # it reproduces the first attempt exactly. Spending it would manufacture a ``stable``
        # verdict out of the recording — and under ``record`` it would additionally overwrite
        # the fixtures the first attempt just laid down.
        return _RetryRefusal('replay', f'not retried: PLURAI_REPLAY={mode} makes a second sample a replay of the first')
    if not budget.claim(scenario_id, reason):
        why = budget.shed_reasons[scenario_id]
        # The counts ride only with the shed that was decided on them; the pool running dry is
        # not a statement about the apparatus and must not dress itself in one.
        return _RetryRefusal(
            why, budget.shed.get(scenario_id, 'not retried'), signals=budget.signal_counts() if why in ('infrastructure', 'wholesale_reds') else {}
        )
    return None


def _clear_stale_retry_dir(scenario_out: Path) -> str | None:
    """Remove any earlier invocation's ``stability_retry/`` before this scenario runs.

    Returns the failure SENTENCE when the directory will not go, else ``None``.

    Never raises: artifact housekeeping must not fail a scenario. A ``logger.warning`` alone is
    not enough: by this module's own text the surviving directory "may be read as THIS run
    second sample" — i.e. another run's artifacts standing in for a measurement of
    this one, which is exactly the class of claim the stability ledger exists to keep honest. So
    the failure travels back to the caller and is filed against the scenario, the way
    ``_clear_retry_error``'s failure already reaches the ledger row.
    """
    stale = scenario_out / STABILITY_RETRY_DIRNAME
    if not stale.exists():
        return None
    try:
        shutil.rmtree(stale)
    except OSError as exc:
        logger.warning('could not remove the stale %s in %s; it may be read as THIS run second sample', STABILITY_RETRY_DIRNAME, scenario_out, exc_info=True)
        return (
            f"an earlier invocation's {STABILITY_RETRY_DIRNAME}/ could not be removed ({type(exc).__name__}: {exc}), so anything "
            'under it belongs to THAT run and a second sample read from this folder may not be this run at all'
        )
    return None


#: The typed fields of a failed retry's ``error.json`` (:func:`_write_error`) that ride its stability row.
_RETRY_FAILURE_KEYS: tuple[str, ...] = ('exception', 'kind', 'cause', 'route_cause', 'transient', 'timed_out', 'budget_seconds', 'error')


def _retry_failure(retry_dir: Path) -> dict[str, Any]:
    """What killed a stability retry, read off its ``error.json`` BEFORE :func:`_clear_retry_error` removes it.

    That file is the retry's only failure record, and the clear exists so the failure sweep does not
    count the retry as a second failed scenario. Without this record the row would say
    ``retry_measured_nothing`` beside a sentence pointing at an EMPTY ``stability_retry/``.
    ``{}`` when the retry wrote no readable record; the row is then still ``retry_failed``, untyped.
    """
    record = _read_json_object(retry_dir / 'error.json') or {}
    failure = {key: record[key] for key in _RETRY_FAILURE_KEYS if record.get(key) is not None}
    if isinstance(failure.get('error'), str):
        failure['error'] = failure['error'][:300]
    return failure


def _clear_retry_error(retry_dir: Path) -> bool:
    """Drop any ``error.json`` the stability retry wrote. True when the folder is clean after.

    The retry lives in a subdirectory precisely so it cannot be mistaken for the scenario's own
    result, and the documented failure sweep is ``find … -name error.json`` — so a retry that
    also failed would otherwise be counted as a second failed scenario. Same reasoning, same
    remedy, as ``_drop_stale_error_json`` applies to ``previous_attempt/``. The partial evidence
    beside it goes too, since the record it explains is deleted here; a copy that will not
    go is logged and does not change the answer, because only ``error.json`` feeds the sweep.
    """
    _drop_stale_partial_evidence(retry_dir)
    error_json = retry_dir / 'error.json'
    (retry_dir / ERROR_DIALOG_FILENAME).unlink(missing_ok=True)  # the other half of the record; carries no count
    if not error_json.exists():
        return True
    try:
        error_json.unlink()
    except OSError:
        logger.warning('could not remove %s; the failure sweep may count this stability retry as a scenario failure', error_json, exc_info=True)
        return False
    return True


#: The stability row's sentence for a ``judge_replay`` second sample, beside its verdict.
_REPLAY_DETAIL = (
    "the first attempt stays authoritative; the second sample re-judged the first attempt's OWN transcript "
    '(POST /threads/{id}/rescore: the same eval set over the same agent output, no agent re-run), so no '
    f'{STABILITY_RETRY_DIRNAME}/ is written'
)


async def _replay_first_attempt(settings: Settings, *, first: ThreadResult, scenario_out: Path) -> tuple[ThreadResult | None, str | None]:
    """The ``judge_replay`` second sample (:data:`SampleKind`): the first attempt's saved
    transcript re-judged by its own thread's eval set, over the backend's replay route.

    The input is the three artifacts the runner just wrote for the first attempt, VERBATIM —
    ``transcript.json`` is the shape the scorer reconstructs the thread from — submitted against the
    first attempt's own thread, so the eval set and the scenario ground truth are the server's to
    resolve, never this client's. Polled on the finalize poll's interval and wallclock budget.

    The replay's ``policy_evaluation_results`` subset replaces the first attempt's on a COPY of it;
    every other field (the dialog, ``ended_early``, the world) stays the first attempt's, because
    that is the input the judges re-read. So the two comparison functions run unchanged over a
    pair whose only moving part is the judge.

    ``(None, why)`` when the replay measured nothing — the route absent on an older backend, the
    job failed, the poll ran out, an artifact unreadable. The row then reads ``unmeasured`` /
    ``retry_measured_nothing`` with ``why`` in its detail. There is NO fallback to an agent re-run:
    that answers a different question, at many times the cost, under a row that would claim it.

    The invocation's BACKEND gate overrules ride the submit like a finalize's
    (:func:`~plurai.orchestrator.client.submit_carrying_overrules`); the replay's ``gate_authority``
    answer is merged into the invocation's collector and kept out of the replayed row, as is every overrule
    marker on its cells (:func:`~plurai.orchestrator.client.take_gate_authority`).
    """
    try:
        transcript = json.loads((scenario_out / 'transcript.json').read_text(encoding='utf-8'))
        world, tools = (
            (json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None)
            for path in (scenario_out / 'final_world_state.json', scenario_out / 'tool_events.json')
        )
    except (OSError, ValueError) as exc:
        return None, f"the first attempt's saved artifacts could not be read ({type(exc).__name__}: {exc})"
    net = settings.network
    client = build_default_client(settings)
    submitted = False
    try:
        submit = await submit_carrying_overrules(
            lambda overrules: client.rescore_async_submit(
                first.thread_id, transcript=transcript, final_world_state=world, tool_events=tools, overrules=overrules
            ),
            endpoint='/threads/{id}/rescore',
        )
        submitted = True
        deadline = time.monotonic() + net.finalize_poll_max_wallclock_seconds
        while (job := await client.rescore_job_status(first.thread_id, submit.job_id)).status in ('queued', 'running'):
            if time.monotonic() >= deadline:
                return None, f'the replay did not finish within the {net.finalize_poll_max_wallclock_seconds:.0f}s finalize poll budget'
            await asyncio.sleep(net.finalize_poll_interval_seconds)
        if job.status == 'failed' and job_refused_for_credit(job):
            raise CreditRefused.for_job(job, thread_id=first.thread_id, job_id=submit.job_id, route='rescore')
    except CreditRefused as exc:
        # The scenario was graded and keeps its verdict; the refusal stops the run, which the replay cannot outlast.
        _stop_run_if_credit_refused(scenario_out.name, exc, billing_url=_billing_url(settings))
        return None, f'the replay was refused for credit ({exc.credit.code})'
    except (httpx.HTTPError, ValueError) as exc:
        # ``ValueError`` covers both a body that is not JSON (``JSONDecodeError`` — a proxy's page on
        # a 2xx) and one that is JSON of the wrong shape (pydantic's ``ValidationError``). Either is a
        # replay that measured nothing; the scenario already ran and was graded, and an annotation
        # must never turn it into a failed row.
        logger.debug('the replay of %s could not be completed', scenario_out.name, exc_info=True)
        if backend_outage(exc):
            # The scenario keeps its verdict and the run goes on: the replay is an annotation, not a recovery that failed, and
            # the next scenario that needs the backend recovers from the outage or stops the run on its own.
            return None, f'the replay could not be completed: {BACKEND_REFUSED_PHRASE}'
        if (refused := _unavailable_abort_from(scenario_out.name, exc)) is not None:
            # The backend's own typed answer is the same for every scenario: the scenario keeps its verdict and the run stops.
            _request_run_abort(refused)
            return None, f'the replay could not be completed: {BACKEND_REFUSED_PHRASE}'
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 404 and not submitted:
                return None, 'the backend answered 404 to POST /threads/{id}/rescore — a backend that predates the judge replay, or a thread it no longer holds'
            return None, f'the replay route answered HTTP {exc.response.status_code}'
        return None, f'the replay could not be completed ({type(exc).__name__})'
    finally:
        await client.aclose()
    if job.status != 'done' or job.result is None:
        return None, f'the replay job failed on the backend ({job.error.detail if job.error else "no result"})'
    # The invocation's backend overrules rode the submit; their answer goes to the gate record, never into the row.
    replayed = take_gate_authority(job.result).model_dump(exclude={'gate_authority'})
    finalize = first.finalize.model_copy()
    finalize.policy_evaluation_results = replayed
    return dataclasses_replace(first, finalize=finalize), None


async def _drive_scenario_with_stability(
    cfg: _RunConfig,
    settings: Settings,
    sink: ObservabilitySink,
    *,
    scenario_id: str,
    dataset_name: str | None,
    run_name: str | None,
    turn_cap_reruns: dict[str, dict[str, Any]] | None,
    budget: _StabilityBudget | None,
    stability: dict[str, dict[str, Any]] | None,
    release_slot: Callable[[], None],
    landed: list[ThreadResult] | None = None,
) -> ThreadResult | None:
    """One scenario, plus the ONE fresh-sampled retry that says whether its failure is REAL.

    ``release_slot`` frees this row's run-concurrency slot (idempotent). It is called before a
    ``judge_replay`` only: that sample is a server rescore of saved artifacts that never drives the
    agent or reads the world, so the next scenario's dialog may start while it is judged. Every
    path that drives the agent (the first attempt, an ``agent_rerun``, an invalid-run retry) holds it.

    ``landed`` receives the first attempt the moment it is graded (an invalid run is not one). A run abort cancels this whole
    coroutine, second sample included, so the caller reads the first attempt from there: a graded
    scenario keeps its verdict and only its stability annotation is lost.

    THE FIRST ATTEMPT STAYS AUTHORITATIVE. Retrying only failures is statistically biased —
    passes are never re-tested — so "pass if either attempt passed" strictly inflates a suite
    score, and this must never do that. The second attempt annotates; it does not decide.

    ONE exception, and it is not an exception to the rule but to its premise: an INVALID run
    never measured the agent at all, so its retry supersedes it the way a ceiling
    re-run does, and costs nothing from the pool.

    TWO QUESTIONS, TWO SAMPLE KINDS (:data:`SampleKind`), and the row names which one it holds
    (``sample_kind``). "Is this scenario's failure a coin flip of the AGENT's?" needs an agent
    resample — a fresh thread against the real agent — and that is ``agent_rerun``, the only kind
    that can publish ``stable`` (the verdict that licenses a redraw). "Is this red JUDGE noise?"
    needs a frozen input — the first attempt's own transcript re-judged — and that is
    ``judge_replay``. Neither answers the other's question: an agent rerun's marks compare two
    verdicts over two different outputs, and a replay's reproduction corroborates the judge, not
    the failure.

    THE SECOND SAMPLE DECIDES NOTHING; A REPLAY CAN NAME ONE THING. Under ``judge_replay`` a cell
    both readings graded and graded OPPOSITE ways is a judge that contradicted itself on ONE input,
    and the row names it under ``contradicted_policies``. The transcript stays byte-for-byte the
    judge's output: the row is the only record, and the backend turns it into an annotation on the
    cell (which the caller may overrule). Only a PASS is ever named — never a red, whose
    withdrawal would raise the score. Under ``agent_rerun`` nothing is named: opposite verdicts
    over two outputs may both be right.

    So the row reads three ways and they do not overlap: a red that REPRODUCED is a measured
    defect (``stable`` only when the agent was resampled), a red that did not is the scenario
    grain's ``flaky``, and a cell one input was graded both ways on is undecided. A cell no second
    sample was taken of is none of the three — the row's ``stability`` and its typed
    ``shed_reason`` are what say so.
    """
    # An earlier invocation's retry directory would otherwise sit in the folder through this
    # run, reading as THIS run's second sample even when this run recorded no stability row.
    # _displace_prior_attempt knows only the top-level artifacts and previous_attempt/, so the
    # hygiene for this directory lives with the code that creates it.
    stale_retry_dir = _clear_stale_retry_dir(cfg.out / scenario_id)
    if stale_retry_dir is not None and budget is not None:
        # The per-scenario sentence channel, filed BEFORE the first attempt so a later refusal
        # carries it too — `_StabilityBudget.note` keeps both sentences rather than displacing
        # one. Every `_record_stability` call below also passes it, so wherever this scenario
        # lands a row, the row says the folder may hold another run's second sample.
        budget.note(scenario_id, stale_retry_dir)
    result = await _drive_one_scenario(
        cfg, settings, sink, scenario_id=scenario_id, dataset_name=dataset_name, run_name=run_name, turn_cap_reruns=turn_cap_reruns
    )
    reason = _stability_reason(result)
    if landed is not None and result is not None and reason != 'invalid_run':
        # An invalid first attempt measured nothing, so it has no verdict to keep.
        landed.append(result)
    if budget is not None:
        budget.observe(scenario_id, result)
    if result is not None and _measured_stop() is not None:
        # The run was stopped for credit (or capacity) while this attempt was finishing: nothing new starts, so no second sample is taken.
        if reason == 'invalid_run':
            _note_not_run(scenario_id)
            return None
        return result
    # `result is None` is already `reason is None` (a crash earns no retry); spelled out so the
    # attempt every second sample below reads is typed as present.
    if reason is None or result is None:
        return result
    if (refusal := _retry_refusal(cfg, budget, scenario_id, reason)) is not None:
        # Every refusal is RECORDED rather than skipped in silence: ``unmeasured`` is a real
        # state and it is not the same as a stable one, and a missing row would say this
        # scenario was never in doubt.
        _record_stability(
            stability,
            scenario_id,
            reason=reason,
            stability='unmeasured',
            detail=refusal.detail,
            refusal=refusal,
            # None, and passed rather than defaulted: `shed_reason` IS this row's typed reason,
            # and a row carrying both would say the retry was refused and that it ran.
            unmeasured_reason=None,
            # Which reds were never re-sampled — every one of them, by name.
            policy_reproduction=(refused := _policy_reproduction(result, None, reason)),
            # …and what each mark WOULD have been evidence of. Written on a refused retry too:
            # `deterministic` is a fact about the policy, not about the retry, and it is the
            # reading that stops a re-run of one expression being read as corroboration.
            policy_evidence=_policy_evidence(result, None, refused),
            # Nothing was contradicted, because no second sample was taken to contradict it.
            contradicted=frozenset(),
            # …and so there is no kind of second sample to name.
            sample_kind=None,
            first_attempt=result,
            stale_retry_dir=stale_retry_dir,
        )
        return result
    if reason == 'invalid_run':
        # Supersedes: ``_drive_one_scenario`` displaces the first attempt into
        # ``previous_attempt/`` at its own entry, so the invalid run is preserved and the
        # scenario folder's top level describes the attempt that actually measured something.
        second = await _drive_one_scenario(
            cfg, settings, sink, scenario_id=scenario_id, dataset_name=dataset_name, run_name=run_name, turn_cap_reruns=turn_cap_reruns
        )
        still_invalid = _stability_reason(second) == 'invalid_run'
        _record_stability(
            stability,
            scenario_id,
            reason=reason,
            # ``stable`` here would mean "the failure reproduced", and an invalid run has no
            # failure to reproduce — it has no measurement at all. A retry that came back
            # invalid AGAIN is the simulator misbehaving twice, which is a fact about the
            # apparatus and is named as such rather than folded into either verdict.
            stability='unmeasured',
            # Hardcoded beside the hardcoded verdict, for the same reason: this branch never
            # consults `_stability_verdict`, so there is no derivation to ask.
            unmeasured_reason='first_attempt_invalid',
            detail=(
                'the first attempt was INVALID (the simulated user repeated itself), so it was never a measurement; '
                'this retry replaces it and was not charged to the pool'
                + (' — and the retry came back INVALID as well, so this scenario has still measured nothing' if still_invalid else '')
                + (' — and the retry FAILED, so the row reported is the invalid first attempt' if second is None else '')
            ),
            # The attempt that actually flows onward. A crashed retry hands the invalid attempt
            # back (an invalid measurement is more than no artifacts), so claiming 'second'
            # unconditionally would assert something false about the row a reader is looking at.
            authoritative='second' if second is not None else 'first',
            # An INVALID first attempt measured nothing, so it graded nothing this retry could
            # contradict — and this branch names no marks at all.
            contradicted=frozenset(),
            # The retry REPLACES a non-measurement, so it is the first sample, not a second one.
            sample_kind=None,
            first_attempt=result,
            retry_attempt=second,
            stale_retry_dir=stale_retry_dir,
        )
        return second if second is not None else result
    sample_kind = cfg.stability_sample_kind
    if _RESAMPLES_THE_AGENT[sample_kind]:
        # A fresh thread against the real agent, its artifacts kept apart from the first attempt's.
        retry_dir = cfg.out / scenario_id / STABILITY_RETRY_DIRNAME
        second = await _drive_one_scenario(
            cfg, settings, sink, scenario_id=scenario_id, dataset_name=dataset_name, run_name=run_name, turn_cap_reruns=None, artifact_dir=retry_dir
        )
        retry_failure = _retry_failure(retry_dir) if second is None else None
        if retry_failure is None:
            detail = f'the first attempt stays authoritative; the retry is a second sample and its artifacts are under {STABILITY_RETRY_DIRNAME}/'
        else:
            cause = retry_failure.get('kind') or retry_failure.get('exception') or 'no failure record'
            detail = (
                f'the first attempt stays authoritative; the retry FAILED before it was graded ({cause}), so it measured nothing '
                f'and kept no artifacts under {STABILITY_RETRY_DIRNAME}/'
            )
        error_json_cleared = _clear_retry_error(retry_dir)
    else:
        # The slot goes first (a rescore touches neither agent nor world); the replay is still
        # awaited here, so the row's clock and its stability ledger entry include it.
        release_slot()
        second, replay_failure = await _replay_first_attempt(settings, first=result, scenario_out=cfg.out / scenario_id)
        detail = _REPLAY_DETAIL + (f' — and the replay measured nothing: {replay_failure}' if replay_failure else '')
        error_json_cleared = True
        retry_failure = None
    # The per-policy record — each mark, and what that mark is evidence of — is read ONCE and the
    # scenario verdict is derived from both, so the fields on the row are one measurement at two
    # grains rather than two measurements. The evidence comes first because the verdict needs it: a
    # `deterministic` reproduction is not a second sample (see `_stability_verdict`).
    reproduction = _policy_reproduction(result, second, reason)
    evidence = _policy_evidence(result, second, reproduction)
    # …and where two readings of ONE input contradict each other, that cell stops being a DECIDED
    # verdict — which only a `judge_replay` can show: over an `agent_rerun` the two verdicts graded
    # two different outputs and may both be right. The names ride onto the row below — the only
    # record of the contradiction, since the transcript is never rewritten — because the marks and
    # their evidence cannot say it happened: ``retry_only`` + ``deterministic`` is also what an
    # ADDED red looks like.
    contradicted = frozenset() if _RESAMPLES_THE_AGENT[sample_kind] else _contradicted_cells(result, reproduction)
    _record_stability(
        stability,
        scenario_id,
        reason=reason,
        stability=_stability_verdict(second, reason, reproduction, evidence=evidence, sample_kind=sample_kind),
        # …and WHY, when that verdict is `unmeasured`. Derived from the same inputs, so the
        # row's verdict and its reason are one reading rather than two; a retry that FAILED is the
        # typed case of "measured nothing", never folded into it.
        unmeasured_reason=_failed_retry_reason(_unmeasured_reason(second, reason, reproduction, evidence=evidence, sample_kind=sample_kind), retry_failure),
        retry_failure=retry_failure,
        detail=detail,
        policy_reproduction=reproduction,
        policy_evidence=evidence,
        contradicted=contradicted,
        sample_kind=sample_kind,
        first_attempt=result,
        retry_attempt=second,
        error_json_cleared=error_json_cleared,
        stale_retry_dir=stale_retry_dir,
    )
    return result


def _goal_reproduction(first: ThreadResult | None, second: ThreadResult | None) -> tuple[str | None, GoalReproduction]:
    """``(the retry's scenario-verdict status, the goal mark)`` — read off each attempt's own block."""
    before = read_scenario_verdict(first.finalize.policy_evaluation_results) if first is not None else None
    after = read_scenario_verdict(second.finalize.policy_evaluation_results) if second is not None else None
    if before is None or after is None or not (before.conclusive and after.conclusive):
        return (after.status if after is not None else None), 'unmeasured'
    return after.status, ('agreed' if before.status == after.status else 'retry_flipped')


def _record_stability(
    ledger: dict[str, dict[str, Any]] | None,
    scenario_id: str,
    *,
    reason: StabilityReason,
    stability: Stability,
    detail: str,
    authoritative: Literal['first', 'second'] = 'first',
    refusal: _RetryRefusal | None = None,
    unmeasured_reason: UnmeasuredReason | None,
    policy_reproduction: Mapping[str, PolicyReproduction] | None = None,
    policy_evidence: Mapping[str, str] | None = None,
    contradicted: frozenset[str],
    sample_kind: SampleKind | None,
    first_attempt: ThreadResult | None,
    retry_attempt: ThreadResult | None = None,
    error_json_cleared: bool = True,
    stale_retry_dir: str | None = None,
    retry_failure: Mapping[str, Any] | None = None,
) -> None:
    """File one scenario's stability record. Written for EVERY scenario that earned a retry,
    including the ones that did not get one — a missing row reads as "never in doubt".

    ``refusal`` is why the retry was NOT spent, typed (``shed_reason``) and, for the
    infrastructure shed, with the per-signal counts it was decided on (``shed_signals``). The
    sentence in ``detail`` says the same thing in words; the typed fields are what a renderer
    branches on, so it never has to parse the words.

    ``unmeasured_reason`` is the OTHER half of that rule, and it is REQUIRED and defaultless for
    the reason ``first_attempt`` is: ``unmeasured`` is reached from four states, one refused and
    three with the retry already spent. A default would let a later call site drop the wiring and
    write a row indistinguishable from a refusal's, which a renderer reads as an older producer.
    A row carries ``shed_reason`` or ``unmeasured_reason``, never both:
    the refused call site passes ``None`` explicitly, and the write below is the ELSE of the
    refusal branch so the exclusion holds structurally rather than by convention.

    ``policy_reproduction`` is the per-policy grain of ``stability`` (:func:`_policy_reproduction`)
    and is written whenever the row names a red at all — a refused retry included, where every
    entry reads ``unmeasured`` and the value of the map is that it names WHICH reds were never
    re-sampled. An INVALID first attempt has no reds worth naming and passes none.

    ``policy_evidence`` is the same map's other column (:func:`_policy_evidence`): what each mark
    is evidence OF. A `code_eval` reads ``deterministic`` — on a ``judge_replay`` its retry re-runs
    one expression over one saved shape, so `✔ reproduced` there is a tautology that would bless a
    deterministic false red as ``stable`` — and a judge reads its two attempts' agreement as a PAIR, because a lone
    `✔ reproduced` is P(red)² and a 60%-catch of a real defect reads "not reproduced" 40% of the
    time it is right.

    ``contradicted`` names the cells two readings of ONE input graded OPPOSITELY (a
    ``judge_replay`` only — see :data:`SampleKind`) — the third state this row could not
    otherwise express, and the ONLY record of it: the transcript keeps the judge's cells as they
    were. The marks alone cannot say it: a deterministic cell reads ``retry_only`` +
    ``deterministic``, which is indistinguishable from a red the retry merely added, and a judged
    one reads ``agreement 1.00 → 1.00`` — the K-sample agreement WITHIN each attempt — which reads
    as perfect agreement for a cell whose two attempts disagreed.

    PASSED IN, and REQUIRED for the reason ``first_attempt`` is: the caller already computed it
    under the sample-kind gate, and a default would let a call site drop the wiring and publish
    silence. ``frozenset()`` is the honest value wherever no second sample was taken.

    ``sample_kind`` is WHAT the second sample was (:data:`SampleKind`), written as the row's
    ``sample_kind`` so a reader can tell marks over two agent outputs from marks over one — the
    distinction the marks themselves cannot carry. ``None`` where no second SAMPLE was taken (a
    refused retry, or an invalid first attempt's replacement, which is a first sample), and then
    the key is omitted. REQUIRED and defaultless for the reason ``contradicted`` is: a default
    would let a call site drop the wiring and publish rows that say nothing about their input.

    ``first_attempt`` / ``retry_attempt`` are the two samples themselves, and the ONLY things read
    off them here are the eval revision each was graded at (:func:`_marks_comparability`): a
    per-policy mark is a comparison, and a comparison across two eval sets is not one; and, where a
    second sample of the AGENT was taken (``agent_rerun``), each attempt's goal verdict
    (:func:`_goal_reproduction`), which annotates the row and moves no verdict.
    ``first_attempt`` is REQUIRED, defaultless: there is always a first attempt, and a default
    lets a later call site drop the wiring and write a row indistinguishable from an unstamped
    backend's with every test still green.

    ``stale_retry_dir`` is the sibling of ``error_json_cleared``: a hygiene failure BEFORE the
    first attempt rather than after the retry, and the same reason for surfacing it — the verdict
    in this row was read out of a folder that may hold another run's artifacts.

    ``retry_failure`` is what killed an agent re-run that returned no attempt (:func:`_retry_failure`),
    ``None`` wherever the retry ran or none was taken; it pairs with ``unmeasured_reason: retry_failed``.
    """
    if ledger is None:
        return
    row: dict[str, Any] = {'reason': reason, 'stability': stability, 'authoritative_attempt': authoritative, 'detail': detail}
    if refusal is not None:
        row['shed_reason'] = refusal.reason
        if refusal.signals:
            row['shed_signals'] = dict(refusal.signals)
    elif unmeasured_reason is not None:
        row['unmeasured_reason'] = unmeasured_reason
    if sample_kind is not None:
        row['sample_kind'] = sample_kind
    if policy_reproduction:
        row['policy_reproduction'] = dict(policy_reproduction)
        # Beside the marks it qualifies, and written EMPTY included wherever they are: a
        # ``retry_only`` cell with no name here is a red the retry added, and one named here is a
        # pass the replay contradicted — the same mark, opposite readings, and an absent key makes
        # the second read like the first.
        row['contradicted_policies'] = sorted(contradicted)
    if policy_evidence:
        row['policy_evidence'] = dict(policy_evidence)
    row.update(_marks_comparability(first_attempt, retry_attempt, marks=policy_reproduction))
    # An agent rerun only: a replay re-judges the first attempt's transcript and carries no goal verdict.
    if sample_kind is not None and _RESAMPLES_THE_AGENT[sample_kind]:
        retry_status, row['goal_reproduction'] = _goal_reproduction(first_attempt, retry_attempt)
        if retry_status is not None:
            row['retry_scenario_verdict'] = retry_status
    if stale_retry_dir:
        row['stale_retry_dir'] = stale_retry_dir
    # The failed retry's typed cause (:func:`_retry_failure`) — its error.json is cleared below the row.
    if retry_failure is not None:
        row['retry_failure'] = dict(retry_failure)
    if not error_json_cleared:
        row['stale_retry_error_json'] = (
            f'{STABILITY_RETRY_DIRNAME}/error.json could not be removed; the documented failure sweep may count this retry as a second failed scenario'
        )
    ledger[scenario_id] = row


async def _run_one(
    cfg: _RunConfig,
    settings: Settings,
    sink: ObservabilitySink,
    *,
    scenario_id: str,
    dataset_name: str | None = None,
    run_name: str | None = None,
    turn_cap_reruns: dict[str, dict[str, Any]] | None = None,
    timings: dict[str, dict[str, Any]] | None = None,
    stability_budget: _StabilityBudget | None = None,
    stability: dict[str, dict[str, Any]] | None = None,
    release_slot: Callable[[], None],
) -> ThreadResult | None:
    """:func:`_drive_one_scenario`, bracketed by the row's OWN progress + clock.

    Everything here is about a finished or in-flight run being READABLE, and each piece closes a
    hole the run itself could not be asked about:

    * the scenario's directory is created and a ``[start]`` line printed AT ENTRY. Every scenario
      launches concurrently, so if both waited for completion nothing whatsoever would land until
      the first one finished, and a healthy multi-scenario run and a wedged one would show the
      same empty terminal and the same empty ``--out`` for minutes.
    * a heartbeat, for the same reason at a longer horizon.
    * ``started_at`` / ``ended_at`` / ``duration_seconds`` into the strategy's ``timings``
      ledger, so ``run_summary.json`` can answer "which scenario was the slow one" without reading
      artifact FILE MTIMES.

    The clock spans the WHOLE row — every re-drive, every ceiling re-run and the stability retry
    inside it — because the question it answers is what this scenario cost, not what its last
    attempt cost. The per-ATTEMPT stamp lives in each transcript's
    ``metadata.started_at``.

    Under the run's leases (:data:`_RESOURCE_LEASES`) a row that has no copy yet is WAITING, not running: it says
    ``[wait]`` once, its ``[start]`` and heartbeat begin when it first holds its copies (:class:`_RowLeaseClock`),
    ``started_at`` is that moment, ``duration_seconds`` leaves out every wait and ``lease_wait_seconds`` says how long
    they were.
    """
    if _run_abort() is not None:
        # The run was stopped by another scenario: this one never starts.
        _note_not_run(scenario_id)
        return None
    entered_wall = time.monotonic()
    # Stamped ONCE per row start: by ``_begin`` when it runs now, else at the wait's start below (``_begin`` re-stamps it
    # when the row first holds its copies). A second unconditional stamp here shifted every per-attempt clock by one.
    started_at = ''
    scenario_out = cfg.out / scenario_id
    try:
        scenario_out.mkdir(parents=True, exist_ok=True)
    except OSError:
        # The artifact writers mkdir again; an early failure here is a progress-reporting
        # problem, not a reason to refuse a scenario that might still run.
        logger.debug('could not pre-create %s; artifact writes will create it', scenario_out, exc_info=True)
    beats: list[asyncio.Task[None]] = []

    def _begin() -> None:
        nonlocal started_at
        started_at = datetime.now(UTC).isoformat(timespec='seconds')
        # INSIDE the bracket, like every other reporting statement here: a start line that raises
        # (a closed pipe — these runs are always piped) must not be the reason a scenario never ran.
        try:
            print(f'[{_ts()}] [start] {label}')
        except Exception:  # noqa: BLE001 — the announcement is not the run
            logger.debug('could not announce the start of %s', label, exc_info=True)
        # The interval is passed rather than left to the parameter default: a default is bound at
        # import time, so the module constant could only be changed by editing the source.
        beats.append(asyncio.create_task(_heartbeat(label, time.monotonic(), _HEARTBEAT_INTERVAL_SECONDS)))

    label = _row_label(cfg, scenario_id)
    row_clock = _RowLeaseClock(label, _begin) if _RESOURCE_LEASES.get() is not None else None
    if row_clock is None:
        _begin()
    else:
        started_at = datetime.now(UTC).isoformat(timespec='seconds')
    clock_token = _ROW_LEASE_CLOCK.set(row_clock)
    landed: list[ThreadResult] = []
    try:
        driven = await _until_run_abort(
            _drive_scenario_with_stability(
                cfg,
                settings,
                sink,
                scenario_id=scenario_id,
                dataset_name=dataset_name,
                run_name=run_name,
                turn_cap_reruns=turn_cap_reruns,
                budget=stability_budget,
                stability=stability,
                release_slot=release_slot,
                landed=landed,
            )
        )
        if driven is _STOPPED:
            if landed:
                # The abort cut off the second sample of a graded scenario: the first attempt keeps its verdict, and the
                # half-written second sample goes.
                _clear_stale_retry_dir(scenario_out)
                return landed[0]
            # Cancelled before anything was graded: the row measured nothing, which is not the same as failing.
            _note_not_run(scenario_id)
            return None
        return cast('ThreadResult | None', driven)
    finally:
        _ROW_LEASE_CLOCK.reset(clock_token)
        for beat in beats:
            beat.cancel()
            # Awaited so the cancellation is COLLECTED here rather than surfacing later as a
            # "Task exception was never retrieved" during interpreter teardown. ``cancel()`` is a
            # NO-OP on a task that already finished, so a heartbeat that RAISED is still sitting on
            # its exception — and a bare ``await`` would re-raise it FROM THE FINALLY, where it
            # escapes the bracket entirely: the timings below never get written, and the exception
            # REPLACES a perfectly good ThreadResult as this row's outcome. Caught and logged, so a
            # progress line can never decide a scenario's fate (``_heartbeat`` also guards its own
            # print, which is belt and braces for exactly this reason).
            try:
                await beat
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — see above
                logger.warning('the heartbeat task for %s raised; the scenario itself is unaffected.', scenario_id, exc_info=True)
        if timings is not None:
            waited = row_clock.waited if row_clock is not None else 0.0
            timings[scenario_id] = {
                'started_at': started_at,
                'ended_at': datetime.now(UTC).isoformat(timespec='seconds'),
                'duration_seconds': round(time.monotonic() - entered_wall - waited, 1),
                **({'lease_wait_seconds': round(waited, 1)} if row_clock is not None else {}),
            }


@_releases_finalize_hold
async def _drive_one_scenario(
    cfg: _RunConfig,
    settings: Settings,
    sink: ObservabilitySink,
    *,
    scenario_id: str,
    dataset_name: str | None = None,
    run_name: str | None = None,
    turn_cap_reruns: dict[str, dict[str, Any]] | None = None,
    artifact_dir: Path | None = None,
) -> ThreadResult | None:
    """Run one scenario end-to-end in its OWN orchestrator; write its artifacts.

    Returns the ``ThreadResult`` on success, ``None`` on failure (after writing
    ``error.json`` and the partial evidence beside it). This is the SOLE scenario runner +
    artifact writer for BOTH strategies: the native gather (which only needs truthiness — a
    ``ThreadResult`` is truthy, ``None`` falsy) AND the experiment task runner (which needs the
    ``ThreadResult`` itself to build the experiment output).

    Per-scenario orchestrator BY DESIGN: the experiment tasks run on the vendor
    executor's own event loop, and an httpx client must not cross loops — so a fresh
    orchestrator (and client) per scenario is the one lifetime that works for both
    strategies. The shared sink is never closed here (the strategy owns that).

    ``turn_cap_reruns`` is the strategy's LEDGER for ceiling-hit re-runs, filled here because
    this is the only frame that sees both attempts. It cannot ride the returned
    ``ThreadResult``: the run summary is built from those results, and the case worth reporting
    is precisely the one where a second attempt replaced a first — the reader needs to know the
    row is a re-run whichever attempt is being reported.

    ``artifact_dir`` moves this attempt's artifacts OFF the scenario's top level. Used by the
    stability retry, whose whole point is that it must not become the scenario's
    answer: the first attempt stays authoritative, so the second writes into
    ``<scenario>/stability_retry/`` and the top level keeps describing exactly one attempt.
    Defaults to the scenario's own folder, which is every other caller.
    """
    scenario_out = artifact_dir if artifact_dir is not None else cfg.out / scenario_id
    # Re-run hygiene: the folder's TOP LEVEL must describe this attempt and no other — a stale
    # error.json beside a fresh transcript.json makes the documented failure collection
    # (find … -name error.json) report a now-passing scenario as failed. The
    # superseded attempt is MOVED aside rather than deleted, so a re-run that then fails cannot
    # destroy a measurement that had already been taken. See _displace_prior_attempt.
    attempt = _AttemptProvenance(prior=_displace_prior_attempt(scenario_out))
    async with _build_orchestrator(cfg, settings, sink) as orch:
        # Re-drive budget, in-memory and PER SCENARIO. Per scenario because a box
        # restart kills every in-flight scenario at once — a budget shared across
        # the gather would rescue one of them and leave the rest exactly as broken
        # as before. Not persisted, because the natural second attempt at a whole
        # CLI invocation is a human re-running it; the persisted budget
        # is the server's, for jobs.
        redrive_budget = REDRIVE_BUDGET
        redrive: RedriveProvenance | None = None
        # The LANDING budget (what an operator sets) and the internal hang detector derived
        # from it. The budget ends the dialog between turns and the scenario still finalizes;
        # this ceiling cancels the attempt outright, so it must sit above every stage bound on
        # the landing path or it would destroy a scenario that was about to land.
        attempt_budget = _resolve_scenario_budget(cfg, settings)
        attempt_ceiling = derive_scenario_hard_ceiling(attempt_budget, settings.network)
        # A qualification pass does NOT spend the re-drive budget on a stalled turn: the
        # phase exists to fail fast, and a per-turn wall that quietly costs twice its length before
        # it reports has defeated itself. Scoped to that ONE trigger — a box restart still
        # re-drives here, because losing the batch to a box blip is not failing fast.
        allow_turn_timeout_redrive = cfg.run_phase != 'qualification'
        # The ceiling-hit re-run budget (--rerun-on-turn-cap), separate from the re-drive one
        # and spent on a different thing: a re-drive answers "the backend went away", this
        # answers "the conversation was still mid-goal when it ran out of turns". Sharing one
        # budget would let a box restart consume the attempt a truncated dialog needs.
        #
        # NOT spent in a qualification pass, on the same reasoning as ``allow_turn_timeout_redrive``
        # directly above: that phase is the fail-fast gate over a small sample, and a scenario
        # that quietly costs two full conversations before it reports has defeated the point of
        # sampling. The typed reason is still recorded there, so the operator sees the ceiling
        # hit and can re-run the full pass with a bigger budget.
        # A single hop stops on its cut by design: there is no turn cap to re-run it at.
        turn_cap_rerun_budget = 1 if _turn_cap_rerun_allowed(cfg) and not cfg.single_hop(scenario_id) else 0
        attempt_max_turns = cfg.turn_cap
        # The attempt a ceiling re-run is trying to improve on. Held so a re-run that FAILS can
        # give it back instead of throwing away a measurement that was already in hand.
        superseded: _SupersededAttempt | None = None

        def _give_back(failure: BaseException) -> ThreadResult | None:
            """Restore the held attempt when a ceiling re-run failed, else ``None`` for the
            ordinary failure path.

            ONE helper for BOTH of this loop's failure exits: the exception handler and the
            stub-fallback exit (a mock-env blip during attempt 2). A failure exit that skipped it
            would discard a GRADED first attempt and report the row as a failure: the
            "6/7 succeeded, 1 FAILED" misreport the restore exists to prevent.
            """
            if superseded is None:
                return None
            return _restore_superseded_attempt(
                cfg,
                settings,
                scenario_out,
                scenario_id=scenario_id,
                superseded=superseded,
                dataset_name=dataset_name,
                run_name=run_name,
                turn_cap_rerun=attempt.turn_cap_rerun,
                failure=failure,
                ledger=turn_cap_reruns,
            )

        credit_token = f'{cfg.trial.index if cfg.trial is not None else 0}:{scenario_id}'
        # Seconds this scenario has waited for room, and whether the attempt after the latest wait still owes its start line.
        waited_for_room = 0.0
        announce_after_wait = False
        while True:
            # A credit stop is polled by the runner every fraction of a second; an attempt that begins in that gap would
            # open a thread the backend refuses, so it is not begun.
            # An earlier attempt's finalize hold does not cover the next one.
            _release_finalize_hold()
            if _measured_stop() is not None:
                _note_not_run(scenario_id)
                return None
            _CREDIT_WAITS.start(credit_token)
            conversation_ended_ok = False
            watch_admission = asyncio.create_task(
                _watch_admission(
                    orch,
                    credit_token,
                    scenario_id,
                    waited_for_room if announce_after_wait else None,
                    earlier_attempt=getattr(orch, 'turns_reached', None),
                )
            )
            announce_after_wait = False
            # Re-captured per ATTEMPT: this stamp is the scenario's start time, and taking it
            # once at entry would make a re-driven or re-run scenario record the moment its FIRST
            # attempt began.
            started_at = datetime.now(UTC).isoformat(timespec='seconds')
            # The world copy this ATTEMPT leased (``world_state.slots``), ``None`` with one shared world.
            world_slot: int | None = None
            try:
                # The per-SCENARIO ceiling, applied HERE rather than around the whole
                # strategy so it holds on BOTH of them: a ceiling living only inside the
                # Arize experiment driver's task timeout leaves the question of whether a
                # scenario can run forever to the configured observability sink. It
                # bounds ONE attempt (a re-drive gets its own), and it wraps ``orch.run``
                # rather than ``_run_one`` so the orchestrator context below stays alive —
                # the /fail bookkeeping and error.json write in the handler still run.
                try:
                    # THE WORLD COPY, leased per ATTEMPT (a re-drive, a turn-cap re-run and every retry
                    # that calls this function again each take their own) for the child's whole life —
                    # reset, seed, turns, snapshot — and given back when ``orch.run`` ends however it
                    # ends; a ceiling cancel has already closed the child by then. Taken OUTSIDE the
                    # setup slot and the ceiling, so it is never counted as setup or as a hang. Separate
                    # from the run slot ``release_slot`` frees: that one is per ROW. The experiment
                    # strategy's pool (``cfg.world_slots``, under its clamp) is never waited on; the
                    # gathered strategy's (:data:`_RESOURCE_LEASES`) is, and goes back the moment the
                    # child closed (``on_child_closed``), before this attempt's /finalize.
                    lease = cfg.world_slots.lease() if cfg.world_slots is not None else nullcontext(None)
                    with lease as clamped_slot:
                        leases = _RESOURCE_LEASES.get()
                        row_clock = _ROW_LEASE_CLOCK.get()
                        if leases is None:
                            held = _HeldLeases()
                        else:
                            held = await (row_clock.acquire(leases) if row_clock is not None else leases.acquire())
                        try:
                            world_slot = held.world_slot if held.world_slot is not None else clamped_slot
                            # The setup slot is taken OUTSIDE the ceiling (waiting for it is not a hang) and given
                            # back the moment the orchestrator reports setup done, or when this attempt exits.
                            async with _SetupSlot(_SETUP_GATE.get()) as setup:
                                result = await asyncio.wait_for(
                                    orch.run(
                                        scenario_set_id=cfg.scenario_set_id,
                                        scenario_id=scenario_id,
                                        eval_set_id=cfg.eval_set_id,
                                        agent_name=scenario_id,
                                        agent_version=cfg.agent_version,
                                        max_turns=attempt_max_turns,
                                        mock_service_url=cfg.mock_service_url,
                                        dataset_name=dataset_name,
                                        scenario_time_budget=attempt_budget,
                                        # The run's own name on POST /threads, so the backend's per-thread
                                        # experiment is this run (each trial its own) rather than one it mints.
                                        experiment_name=run_name,
                                        run_facts=_scenario_run_facts(cfg, run_name=run_name, scenario_id=scenario_id),
                                        on_setup_done=setup.release,
                                        on_finalize_start=_mark_finalizing,
                                        replay_turn_only=cfg.single_hop(scenario_id),
                                        # Landing a failed turn and re-driving it are exclusive: the dialog lands
                                        # only when this loop would not re-drive the attempt for that failure.
                                        redrive_available=redrive_budget > 0,
                                        allow_turn_timeout_redrive=allow_turn_timeout_redrive,
                                        on_child_closed=held.release,
                                        world_slot=world_slot,
                                    ),
                                    timeout=attempt_ceiling,
                                )
                        finally:
                            held.release()
                except TimeoutError as timeout_exc:
                    raise ScenarioTimeout(
                        f'scenario exceeded its {attempt_ceiling:.0f}s HARD ceiling across all turns and was cancelled '
                        f'(derived from the {attempt_budget:.0f}s time budget + per-turn {settings.network.agent_turn_timeout:.0f}s '
                        f'+ the finalize poll budget). This is a HANG detector, not the time budget: a scenario that merely runs '
                        f'long is ended by the budget between turns and still finalized. Reaching this means a stage bound did not '
                        f'hold — inspect the last turn and the mock rather than raising --scenario-timeout.',
                        budget_seconds=attempt_ceiling,
                        thread_id=orch.thread_id,
                    ) from timeout_exc
                conversation_ended_ok = True
            except asyncio.CancelledError:
                if (abort := _run_abort()) is not None:
                    await _close_attempt_cancelled_by_abort(orch, abort)
                raise
            except Exception as exc:  # noqa: BLE001 — any agent/adapter failure becomes error.json
                # thread_id: the typed attribute when the error carries one (e.g.
                # GraphRecursionWrap), else the id the orchestrator recorded at
                # create — which is what stops error.json from saying "unknown" and
                # lets /fail actually reach a thread that still exists.
                thread_id = (exc.thread_id if isinstance(exc, OrchestratorAdapterError) else None) or orch.thread_id
                # A new conversation the gateway refused for room, while the run holds conversations of its own, waits for one
                # of them to free it and starts afresh: no error record, no re-drive, nothing graded. With none of its own the
                # refusal is someone else's work and stops the run below like any credit refusal.
                if _refused_new_start(exc, orch):
                    watch_admission.cancel()
                    if await _decide_refused_start(credit_token) == 'wait':
                        waited = await _wait_for_credit_room(orch, credit_token, scenario_id, thread_id)
                        if waited is None:
                            _note_not_run(scenario_id)
                            return None
                        waited_for_room += waited
                        announce_after_wait = True
                        continue
                # A mock that cannot answer is the same answer on a re-drive and on every other scenario: stop the run.
                stopped = _stop_run_if_mock_cannot_answer(scenario_id, exc, mock_url=getattr(orch, 'mock_service_url', None) or cfg.mock_service_url)
                # A credit or capacity refusal, or the backend's own typed unavailable answer, is the same answer on a re-drive
                # and for every other scenario: stop the run, and do not re-drive. An outage is recovered first (below).
                if (measured_stop := _stop_run_if_refused(scenario_id, exc, billing_url=_billing_url(settings))) is not None:
                    # It measured nothing, and the fault is the organization's credit or the backend, not the agent: a row that
                    # was not run.
                    _note_not_run(scenario_id)
                    _hold_through_bookkeeping()
                    stopped = stopped or measured_stop
                    if isinstance(measured_stop, _UnavailableAbort):
                        exc = BackendUnavailable(UnavailableBlock(measured_stop.cause))
                # A run that already stopped re-drives nothing: a finalize poll the stop spared and that failed after it
                # would otherwise wait out the readiness gate before the stop's text prints.
                if redrive_budget > 0 and stopped is None and _run_abort() is None:
                    # An EMPTY WORLD at the start of the dialog is a seed race, re-driven whatever class the runtime
                    # gave its fault; ``getattr``: a duck-typed orchestrator need not carry the attempt's facts.
                    empty_world = seed_fault(
                        exc,
                        turns_completed=len(getattr(orch, 'turns_reached', [])),
                        seeded_entities=getattr(orch, 'seeded_entities', 0),
                        served_data=_pre_crash_tool_activity(getattr(orch, 'progress', None)) == 'real_data',
                    )
                    trigger = (
                        RedriveTrigger.SEED_FAULT
                        if empty_world
                        else await classify_restart_failure(
                            exc,
                            client=orch.client,
                            thread_id=thread_id,
                            restart_shaped_calls=0,
                            allow_turn_timeout_redrive=allow_turn_timeout_redrive,
                            concurrency=_sibling_width(cfg),
                        )
                    )
                    try:
                        ready = trigger is not None and await _gate_for_redrive(
                            cfg, settings, orch, scenario_id=scenario_id, trigger=trigger, thread_id=thread_id
                        )
                    except asyncio.CancelledError:
                        # The run stopped while this scenario waited for the backend: its thread is closed like any other the
                        # stop cancels, or it holds the organization's room until the gateway's idle timeout.
                        if (abort := _run_abort()) is not None:
                            await _close_attempt_cancelled_by_abort(orch, abort)
                        raise
                    if ready and trigger is not None:
                        redrive_budget -= 1
                        # An outage's own text names a status, a URL or an id: the provenance records the cause alone.
                        shown = BackendUnavailable(UnavailableBlock(block.cause)) if backend_outage(exc) and (block := unavailable_block_of(exc)) else exc
                        redrive = RedriveProvenance(
                            original_thread_id=thread_id,
                            trigger=trigger,
                            original_error=f'{type(shown).__name__}: {shown}',
                            original_kind=getattr(type(exc), 'kind', None),
                            original_transient=getattr(exc, 'transient', None),
                        )
                        await orch._cancel_inflight_finalize()
                        continue
                if stopped is None and (outage_stop := _stop_run_for_outage(scenario_id, exc)) is not None:
                    # Recovery did not cure the outage: the backend stayed down past the readiness gate, the re-drive met it
                    # again, or the re-drive was already spent. Every other scenario would meet it too.
                    _note_not_run(scenario_id)
                    _hold_through_bookkeeping()
                    stopped = outage_stop
                    exc = BackendUnavailable(UnavailableBlock(outage_stop.cause))
                elif stopped is None and isinstance(exc, MockRuntimeFaultRaised) and unavailable_block_of(exc) is not None:
                    # A hosted mock's unavailable answer the re-drive did not cure: this scenario was not completed, through
                    # no fault of the agent, and the run goes on.
                    _note_not_run(scenario_id)
                # Stop any in-flight async finalize job to avoid billing LLM judges after the SDK has already given up on
                # this scenario. The transport's own text names a status, a URL or an id, so an outage's ``/fail`` carries
                # :class:`BackendUnavailable` alone; a mock fault's words are the SDK's own and already say nothing of either.
                await _close_thread(orch, thread_id, exception=type(exc).__name__, error=str(exc))
                # A ceiling re-run that FAILED gives back the attempt it was trying to improve
                # on: that attempt is a real measurement, already graded, and reporting this row
                # as a failure would delete it from graded_scenarios, ended_early_scenarios and
                # measurement_integrity alike. The optional re-run is a bet the runner may lose
                # without costing what it already had.
                restored = _give_back(exc)
                if restored is not None:
                    return restored
                if isinstance(exc, HarnessIpcOverrun):
                    _record_ipc_overrun(scenario_id, exc)
                # ``getattr``: this is the failure path and must not raise, and test doubles of
                # Orchestrator need not carry ``progress``. Read once and shared with
                # ``_write_error`` (``pre_crash_tool_activity``) so the two never disagree
                # about which attempt's tool events they describe.
                progress = getattr(orch, 'progress', None)
                record = _write_error(
                    scenario_out,
                    scenario=scenario_id,
                    thread_id=str(thread_id or 'unknown'),
                    exc=exc,
                    redrive=redrive,
                    attempt=attempt,
                    progress=progress,
                    # A duck-typed orchestrator may not carry it; the real one always does.
                    turns_reached=list(getattr(orch, 'turns_reached', [])),
                    world_slot=world_slot,
                    run_context=_run_context(
                        cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=started_at, max_turns=attempt_max_turns, world_slot=world_slot
                    ),
                )
                _write_partial_evidence(scenario_out, scenario=scenario_id, progress=progress)
                print(f'[{_ts()}] {_fail_line(_row_label(cfg, scenario_id), record)}', file=sys.stderr)
                return None
            finally:
                watch_admission.cancel()
                _CREDIT_WAITS.end(credit_token, finalized=conversation_ended_ok)

            # TOOL CALLS THE MOCK NEVER ANSWERED (the agent was served a stub). A measurement fact,
            # never a verdict: the backend already graded this dialog, the world was read at finalize
            # (which is where world trust is decided), and the row is graded with a finding. The one
            # thing these calls still decide is RETRY POLICY: calls that failed the way a mock
            # restart fails mean the seeded session is gone, and a re-drive is the only way back to a
            # full measurement.
            unanswered = result.unanswered_mock_calls or []
            if unanswered and (measured_stop := _stop_run_if_refused(scenario_id, unanswered, billing_url=_billing_url(settings))) is not None:
                # The mock's model call was refused for credit or capacity and the dialog was graded against stubs: no verdict
                # of it is kept, and the run stops. The row is one that was not run, with an error record naming the refusal.
                _note_not_run(scenario_id)
                credit_block = CreditBlock(measured_stop.code, measured_stop.resets_at) if isinstance(measured_stop, _CreditAbort) else None
                capacity_block = CapacityBlock(measured_stop.resets_at, measured_stop.reset_source) if isinstance(measured_stop, _CapacityAbort) else None
                refusal = measured_stop.route_cause
                shown_cause = refusal if refusal == 'credit' else BACKEND_REFUSED_PHRASE  # a capacity stop never names it
                fault = MockRuntimeFaultRaised(
                    f'mock runtime fault: {len(unanswered)} tool call(s) went unanswered by the mock ({shown_cause})',
                    transient=False,
                    cause=measured_stop.remedy,
                    route_cause=refusal,
                    credit=credit_block,
                    capacity=capacity_block,
                    child_exc_class='plurai.MockRouteFault',
                    thread_id=result.thread_id,
                )
                record = _write_error(
                    scenario_out,
                    scenario=scenario_id,
                    thread_id=str(result.thread_id or 'unknown'),
                    exc=fault,
                    redrive=redrive,
                    attempt=attempt,
                    progress=getattr(orch, 'progress', None),
                    turns_reached=list(getattr(orch, 'turns_reached', [])),
                    world_slot=world_slot,
                    run_context=_run_context(
                        cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=started_at, max_turns=attempt_max_turns, world_slot=world_slot
                    ),
                )
                print(f'[{_ts()}] {_fail_line(_row_label(cfg, scenario_id), record)}', file=sys.stderr)
                return None
            if (
                unanswered
                and (
                    unanswerable := _stop_run_if_mock_cannot_answer(
                        scenario_id, unanswered, mock_url=getattr(orch, 'mock_service_url', None) or cfg.mock_service_url
                    )
                )
                is not None
            ):
                # The dialog was graded against stubs for tools that never worked: no verdict of it is kept, and the
                # run stops. The row keeps an error record naming the cause.
                fault = MockRuntimeFaultRaised(
                    f'mock runtime fault: {len(unanswered)} tool call(s) went unanswered by the mock ({unanswerable.cause})',
                    transient=False,
                    cause=unanswerable.remedy,
                    route_cause=unanswerable.cause,
                    child_exc_class='plurai.MockRouteFault',
                    thread_id=result.thread_id,
                )
                record = _write_error(
                    scenario_out,
                    scenario=scenario_id,
                    thread_id=str(result.thread_id or 'unknown'),
                    exc=fault,
                    redrive=redrive,
                    attempt=attempt,
                    progress=getattr(orch, 'progress', None),
                    turns_reached=list(getattr(orch, 'turns_reached', [])),
                    world_slot=world_slot,
                    run_context=_run_context(
                        cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=started_at, max_turns=attempt_max_turns, world_slot=world_slot
                    ),
                )
                print(f'[{_ts()}] {_fail_line(_row_label(cfg, scenario_id), record)}', file=sys.stderr)
                return None
            if unanswered and (refused_unavailable := _unavailable_abort_from(scenario_id, unanswered)) is not None:
                # The backend answered a mocked call as unavailable (a provider blip behind one tool) and the dialog was graded
                # against stubs: no verdict of it is kept. Like a transient mock fault the scenario is re-driven once; a row the
                # re-drive did not cure was not completed, through no fault of the agent, and the run goes on.
                fault = MockRuntimeFaultRaised(
                    f'mock runtime fault: {len(unanswered)} tool call(s) went unanswered by the mock ({BACKEND_REFUSED_PHRASE})',
                    transient=False,
                    cause=BACKEND_REFUSED_PHRASE,
                    route_cause='unavailable',
                    unavailable=UnavailableBlock(refused_unavailable.cause),
                    child_exc_class='plurai.MockRouteFault',
                    thread_id=result.thread_id,
                )
                trigger = RedriveTrigger.MOCK_RUNTIME_FAULT
                if (
                    redrive_budget > 0
                    and _run_abort() is None
                    and await _gate_for_redrive(cfg, settings, orch, scenario_id=scenario_id, trigger=trigger, thread_id=result.thread_id)
                ):
                    redrive_budget -= 1
                    redrive = RedriveProvenance(
                        original_thread_id=result.thread_id,
                        trigger=trigger,
                        original_error=str(fault),
                        original_kind=MockRuntimeFaultRaised.kind,
                        original_transient=False,
                    )
                    continue
                _note_not_run(scenario_id)
                record = _write_error(
                    scenario_out,
                    scenario=scenario_id,
                    thread_id=str(result.thread_id or 'unknown'),
                    exc=fault,
                    redrive=redrive,
                    attempt=attempt,
                    progress=getattr(orch, 'progress', None),
                    turns_reached=list(getattr(orch, 'turns_reached', [])),
                    world_slot=world_slot,
                    run_context=_run_context(
                        cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=started_at, max_turns=attempt_max_turns, world_slot=world_slot
                    ),
                )
                print(f'[{_ts()}] {_fail_line(_row_label(cfg, scenario_id), record)}', file=sys.stderr)
                return None
            if unanswered:
                restart_shaped = restart_shaped_calls(unanswered)
                # The re-drive's provenance carrier and the give-back's stated cause, never raised.
                stub_error = MockStubFallback(
                    f'{len(unanswered)} tool call(s) went unanswered by the mock ({restart_shaped} the way a mock restart fails)',
                    thread_id=result.thread_id,
                )
                if restart_shaped and redrive_budget > 0:
                    trigger = await classify_restart_failure(
                        stub_error,
                        client=orch.client,
                        thread_id=result.thread_id,
                        restart_shaped_calls=restart_shaped,
                        allow_turn_timeout_redrive=allow_turn_timeout_redrive,
                        concurrency=_sibling_width(cfg),
                    )
                    if trigger is not None and await _gate_for_redrive(
                        cfg, settings, orch, scenario_id=scenario_id, trigger=trigger, thread_id=result.thread_id
                    ):
                        redrive_budget -= 1
                        redrive = RedriveProvenance(
                            original_thread_id=result.thread_id,
                            trigger=trigger,
                            original_error=str(stub_error),
                            original_kind=MockStubFallback.kind,
                            original_transient=getattr(stub_error, 'transient', None),
                        )
                        continue
                _record_unanswered_calls(scenario_id, unanswered)
                # A ceiling re-run whose calls went unanswered gives back the attempt it was betting
                # against: that attempt measured with every call answered, this one did not.
                restored = _give_back(stub_error)
                if restored is not None:
                    return restored

            # How the dialog LANDED — a turn failure it was graded to, a snapshot channel left unread,
            # world values carried in their JSON form — recorded as findings, never as a failure.
            result = _with_landing_evidence(result, getattr(orch, 'progress', None))
            _record_landing_findings(scenario_id, result)
            # Written INSIDE the orchestrator scope so a teardown failure can't lose them.
            _write_success_artifacts(
                scenario_out,
                scenario=scenario_id,
                result=result,
                run_context=_run_context(
                    cfg, settings, dataset_name=dataset_name, run_name=run_name, started_at=started_at, max_turns=attempt_max_turns, world_slot=world_slot
                ),
                redrive=redrive,
                attempt=attempt,
            )
            # CEILING-HIT RE-RUN (opt-in). The conversation ran out of turns while the user
            # simulator's own per-turn verdict still said the target was unmet: its evals are
            # low for obligations the dialog never reached, and reading that as the agent's
            # failure is a misread. The re-run automates what an operator would do by hand:
            # re-run the scenario at a raised budget and compare.
            #
            # Both conditions are required, and both are TYPED: the harness's own reason for
            # ending (never a turn count) and the backend's own target verdict (never inferred
            # from the transcript). Artifacts are written FIRST and displaced after, so the
            # first attempt is recorded whatever the second one does.
            if turn_cap_rerun_budget > 0 and result.ended_early == TURN_CAP_REASON and _run_abort() is None:
                target = _target_status(result)
                if target != 'incomplete':
                    _log_turn_cap_rerun_declined(scenario_id, target)
                    break
                turn_cap_rerun_budget -= 1
                rerun = _TurnCapRerunProvenance(
                    previous_thread_id=result.thread_id,
                    previous_max_turns=attempt_max_turns,
                    max_turns=attempt_max_turns * TURN_CAP_RERUN_MULTIPLIER,
                )
                # Attempt 2's own provenance, replacing attempt 1's wholesale: the artifacts
                # just written are displaced under previous_attempt/, and the re-drive that may
                # have rescued attempt 1 says nothing about attempt 2 — leaving it set would make
                # the re-run's artifacts claim a re-drive that never happened to them.
                attempt = _AttemptProvenance(prior=_displace_prior_attempt(scenario_out), turn_cap_rerun=rerun)
                redrive = None
                attempt_max_turns = rerun.max_turns
                # The TIME budget scales with the turn budget it is bounding, and the ceiling is
                # re-derived from it. Neither knows about max_turns, so a doubled conversation
                # held to the original clock would be cut at the same wall-clock second as the
                # attempt it is meant to extend — converting a turn-cap re-run into a
                # time-budget landing and silently disabling the feature. A scenario engineered
                # to hit the wall instead of the cap is a worse failure than the one being avoided.
                attempt_budget = attempt_budget * TURN_CAP_RERUN_MULTIPLIER
                attempt_ceiling = derive_scenario_hard_ceiling(attempt_budget, settings.network)
                # Held so a re-run that fails can hand back the measurement it was trying to
                # improve on, rather than turning a graded scenario into a failed row. The
                # stamps ride along because the loop's own are about to describe attempt 2.
                superseded = _SupersededAttempt(result=result, started_at=started_at, max_turns=rerun.previous_max_turns)
                # The strategy's ledger, so the run summary can say this row is a re-run and
                # whether the attempt it replaced had already produced a verdict. Recorded
                # BEFORE attempt 2 runs: whatever happens to it, the fact that a re-run
                # happened — and what was given up for it — is already written down.
                if turn_cap_reruns is not None:
                    turn_cap_reruns[scenario_id] = {
                        **rerun.to_artifact(),
                        'previous_attempt_graded': _scenario_was_graded(result) and cfg.is_scored,
                        'previous_attempt_artifacts': attempt.prior.preserved,
                    }
                _announce_turn_cap_rerun(scenario_id, rerun)
                continue
            break
    # ``[ok]`` means the scenario RAN. When nothing about it was graded the pass_rate and
    # aggregate_score on this very line are 0.0 defaults over zero landed policies, not
    # verdicts, so the tag says so rather than leaving a clean-looking 0.000 to be read as a
    # score. Keyed on the finalize the line is about, not on ``--eval-set-id``: a scenario
    # whose judges all errored prints the same 0.000 as one that was never graded at all.
    # See _print_summary for the run-level counterpart.
    tag = '[ok]  ' if _scenario_was_graded(result) and cfg.is_scored else '[ok/unscored]'
    print(f'[{_ts()}] {tag} {_row_label(cfg, scenario_id)}: {result.summary()}')
    return result


async def _gate_for_redrive(
    cfg: _RunConfig,
    settings: Settings,
    orch: Orchestrator,
    *,
    scenario_id: str,
    trigger: RedriveTrigger,
    thread_id: str | None,
) -> bool:
    """Prove the backend can serve a fresh scenario, then announce the re-drive.

    Returns ``True`` when the attempt should be spent. ``False`` means readiness
    never arrived within the gate's bound (or the mock is held on a human), in
    which case the caller falls through to the ORIGINAL error and the budget goes
    unspent — a re-drive into a backend that is still coming up would just repeat
    the failure and consume the one attempt we have.

    Every trigger passes through here, including the ones that do NOT imply a dead
    backend (a stalled agent turn): the gate is cheap when nothing is wrong — one
    ``/health`` plus one mock probe — and it is the single place that decides a
    re-drive is worth its cost.
    """
    # The URL the SERVER handed this scenario, when create got that far; the CLI's
    # --mock-service-url is only the fallback (it is routinely omitted).
    mock_url = orch.mock_service_url or cfg.mock_service_url
    if not await wait_until_ready(orch.client, mock_service_url=mock_url, mock_mode=settings.network.mock_mode):
        logger.warning(
            'scenario %s failed in a re-drivable way (%s, thread %s), but the backend did not '
            'become ready within the readiness gate — surfacing the original error; the re-drive budget is unspent.',
            scenario_id,
            trigger.value,
            thread_id or 'unknown',
        )
        return False
    logger.warning(
        're-driving scenario %s as a NEW thread: thread %s was lost to a transient failure (%s). This is a full re-run of the scenario, billed as a new run.',
        scenario_id,
        thread_id or 'unknown',
        trigger.value,
    )
    if trigger is RedriveTrigger.BACKEND_OUTAGE:
        # A customer reads this line during an outage: the cause in plain words, with no id, status or trigger name.
        print(f'[{_ts()}] [redrive] {scenario_id}: the backend stopped answering; re-running the scenario once.', file=sys.stderr)
    else:
        print(f'[{_ts()}] [redrive] {scenario_id}: thread {thread_id or "unknown"} was lost ({trigger.value}); re-running the scenario once.', file=sys.stderr)
    return True


async def _upload_datasets(cfg: _RunConfig, settings: Settings, sink: ObservabilitySink, *, scenario_ids: Sequence[str] | None = None) -> UploadResult | None:
    """Best-effort: fetch the set + upload it to the active client-side sink.

    Runs as a PRE-FLIGHT, before the gather — the dataset is the experiment's INPUT,
    and a sink that pins its experiment session to a dataset version needs every
    example to exist first (see the call site in ``_run_gathered_async``). Triggered
    by ``--upload-datasets`` OR by a sink declaring ``requires_preflight_dataset``.
    ``scenario_ids`` limits the upload to the scenarios this run executes.
    Gated on client sink keys (no keys ⇒ a graceful no-op). The upload itself never
    aborts the run — a transport / sink failure is reported as a status line, not
    raised. We print exactly one per-provider result line so the operator sees what
    happened.

    Opens its OWN short-lived orchestrator (mirroring ``_setup_experiment``); the
    upload goes to the SAME single active sink the whole run holds, so the CLI
    still owns env→sink selection.

    Returns the :class:`UploadResult`, or ``None`` when the fetch/upload raised. The
    caller needs BOTH axes and a bare ``ok`` cannot carry them: ``ok`` drives the exit
    code (a lossy upload must not exit 0), while ``uploaded`` distinguishes "uploaded
    everything" from "uploaded nothing" — which is ``ok=True`` but is still a broken
    pre-flight for a sink whose experiment requires a populated dataset.
    """
    try:
        async with _build_orchestrator(cfg, settings, sink) as orch:
            result = await orch.upload_datasets(scenario_set_id=cfg.scenario_set_id, out_dir=cfg.out, scenario_ids=scenario_ids)
    except Exception as exc:  # noqa: BLE001 — a fetch error must not crash the runner before the scenarios even start
        print(f'[dataset-upload] FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return None
    print(result.result_line())
    return result


@asynccontextmanager
async def experiment_scope(sink: ObservabilitySink, *, dataset_name: str, run_name: str) -> AsyncIterator[None]:
    """Bracket a scenario-SET gather as ONE Experiment / Dataset-Run (optional, best-effort seam).

    Async context manager that opens the set-level experiment on ``__aenter__`` and closes
    it on ``__aexit__`` for sinks that implement the optional
    :class:`~plurai.observability.sink_protocol.ExperimentLifecycleSink` seam. A sink that
    does not model set-level experiments is simply not an ``ExperimentLifecycleSink`` and
    is skipped (purely additive — no existing per-scenario behavior changes).

    Two best-effort guarantees:
    * ``begin_experiment`` failure ⇒ logged to stderr AND the seam is dropped, so
      ``end_experiment`` is NEVER called for an experiment that never opened; and
    * ``end_experiment`` runs from a ``finally`` ⇒ it fires even when the wrapped gather
      RAISES — closing the experiment instead of leaking it. (The dataset upload runs
      ahead of the gather, since the dataset defines the experiment's inputs.)

    The Protocol stays a TWO-METHOD contract (``begin_experiment`` / ``end_experiment``):
    ``post_feedback`` reads the run handle off ``self`` mid-gather, so the lifecycle can't
    live in a CM frame on the sink — the context manager is purely a CALLER-side bracket.

    Best-effort throughout: a seam failure can NEVER abort the run.
    """
    seam = sink if isinstance(sink, ExperimentLifecycleSink) else None
    if seam is not None:
        try:
            await seam.begin_experiment(dataset_name=dataset_name, run_name=run_name)
        except Exception as exc:  # noqa: BLE001 — experiment lifecycle is best-effort; never abort the run
            print(f'[experiment] begin_experiment failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            seam = None  # don't end what never began
    try:
        yield
    finally:
        if seam is not None:
            try:
                await seam.end_experiment()
            except Exception as exc:  # noqa: BLE001 — experiment lifecycle is best-effort; never abort the run
                print(f'[experiment] end_experiment failed: {type(exc).__name__}: {exc}', file=sys.stderr)


class _ExperimentSink(ExperimentRunnerSink, ObservabilitySink, Protocol):
    """Typing-only intersection view: the experiment strategy drives BOTH roles.

    ``main()`` narrows via ``isinstance(sink, ExperimentRunnerSink)`` — the narrowed
    value satisfies this structurally, so the strategy can call ``run_experiment_set``
    AND the ordinary sink lifecycle (``warm_once`` / ``close``) without casts.
    """


def _declared_agent_identity(workspace_root: Path) -> str | None:
    """``<workspace>/<project>@<agent.version>`` from ``plurai.config.yaml``, or ``None``.

    THE DECLARED identity, preferred over anything derived, because a derived one describes the
    wrong subject whenever the harness lives inside another repo. A bench fixture staged inside the
    backend's repository, for example, would be stamped with the backend's own ``git describe``,
    filed as the agent's build.

    ``@<version>`` only when ``agent.version`` is declared; ``workspace``/``project`` alone are a
    complete identity for an agent that does not version itself. An ABSENT config is ``None`` and
    silent — nothing was declared, which is the ordinary state.

    A config that EXISTS and cannot testify is a different fact and is SAID. ``discover_config_
    documents`` raises for a present config that is unreadable, unparseable or not a mapping, and
    that raise is a deliberate fail-closed: nothing else in the SDK reports it, and swallowing it
    at INFO (below the default level) would turn it into silence, so a typo'd
    ``plurai.config.yaml`` would ship a run stamped with whatever repo the process stood in. It
    does not STOP the run — this runs on the way to a required wire field, and there is a derived
    tag to fall back to — but it degrades where the operator can see it.

    ``OSError`` is caught beside it, so that the promise above holds. ``discover_config_documents`` calls ``Path.exists()`` and
    ``is_file()`` BEFORE the guarded ``read_text``, and ``Path.exists()`` re-raises ``EACCES``
    (only ENOENT/ENOTDIR/EBADF/ELOOP are ignored). So a ``plurai_sim/`` at mode ``000`` — a
    permissions accident, or a harness dir owned by another user — raises ``PermissionError``
    straight through a narrower ``except``, through every frame up to ``main()``, and reaches the
    operator as a raw traceback BEFORE any artifact is written. That turns a run
    that can survive into a run that dies, on the way to a field it has a fallback for.
    """
    try:
        documents = discover_config_documents(workspace_root)
    except (WorkspaceConfigError, OSError) as exc:
        logger.warning('--agent-version not supplied and %s could not be read (%s); falling back to the derived tag.', CONFIG_FILENAME, exc)
        print(
            f'[agent-version] WARNING: {CONFIG_FILENAME} exists and could not be read ({exc}), so this run is stamped '
            f'with a DERIVED tag naming whatever repository it stands in — not the agent identity the config declares. '
            f'Fix the config, or pass --agent-version.',
            file=sys.stderr,
        )
        return None
    for _path, document in documents:
        workspace = str(document.get('workspace') or '').strip()
        project = str(document.get('project') or '').strip()
        if not workspace or not project:
            continue
        agent = document.get('agent')
        version = str(agent.get('version') or '').strip() if isinstance(agent, Mapping) else ''
        return f'{workspace}/{project}@{version}' if version else f'{workspace}/{project}'
    return None


def _described_repo_tag(workspace_root: Path) -> str:
    """``<repo>@<git describe>`` — the DERIVED tag, naming the repo it described.

    ``git describe`` alone answers a question nobody asked: it names whatever repository the
    process happens to stand in, and the artifact then claims that tag as the AGENT's build. The
    repository name is what makes the claim auditable — ``other-repo@v0.3.0-50-dirty`` is
    obviously not the agent under test, while the bare tag beside it is not obviously anything.
    """

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell, no user input
                ['git', *args],  # noqa: S607 — git resolved via PATH by design
                cwd=workspace_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                **utf8_child_text(),  # git prints UTF-8
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.info('`git %s` failed while deriving an agent tag (%s).', args[0], exc)
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    described = _git('describe', '--tags', '--always', '--dirty')
    if not described:
        # Operator-VISIBLE degradation: every observability record gets tagged
        # ``agent_version="unknown"``. A breadcrumb so a wall of "unknown" tags is explainable
        # (git not on PATH / not a repo / no commits / timeout) rather than a silent mystery.
        logger.info('--agent-version not supplied and `git describe` produced nothing; tagging this run as "unknown".')
        return 'unknown'
    toplevel = _git('rev-parse', '--show-toplevel')
    repo = Path(toplevel).name if toplevel else ''
    return f'{repo}@{described}' if repo else described


def _default_agent_version(workspace_root: Path | None = None) -> str:
    """The agent-build tag to stamp when the operator passes no ``--agent-version``.

    DECLARED first (``plurai.config.yaml``), derived second — because the derived tag names the
    repository the runner stands in, which is the agent's own repo only when the harness is not a
    fixture inside someone else's. REQUIRED on the wire, so this NEVER returns ``None`` or ``''``:
    the last rung is the literal ``"unknown"``, so the create-thread call cannot 422 on a missing
    version.
    """
    root = workspace_root or Path.cwd()
    return _declared_agent_identity(root) or _described_repo_tag(root)


@dataclass(frozen=True, slots=True)
class _TrialIdentity:
    """Which trial of a ``--trials N`` invocation a trial body runs: ``index`` of ``count``, and
    the ``group`` — the invocation's base run name — every trial of it shares."""

    index: int
    count: int
    group: str


class _WorldSlots:
    """The free list of an external or agent-owned world's copies (``world_state.slots: N``): indices ``0..N-1``.

    Each scenario ATTEMPT leases one copy for its child's whole life (:meth:`lease`, around
    ``orch.run`` in :func:`_drive_one_scenario`) and gives it back when that attempt ends, however it
    ends. Never awaited: the run is clamped to ``concurrency = min(C, N)`` with trials in turn
    (:func:`_one_at_a_time_for_world`), so attempts in flight never outnumber copies, and a lease
    from an EMPTY pool is a bug that raises — into that scenario's ``error.json`` — rather than waits.

    A ``threading.Lock``, not an asyncio primitive and not a ContextVar: it lives on
    :class:`_RunConfig` so the experiment strategy's vendor-executor threads (which see neither the
    gathered loop nor its ContextVars) lease from the same pool.
    """

    def __init__(self, count: int) -> None:
        self.count = count
        self._lock = threading.Lock()
        self._free: deque[int] = deque(range(count))

    @property
    def free(self) -> int:
        """How many copies are not leased right now."""
        with self._lock:
            return len(self._free)

    @contextmanager
    def lease(self) -> Iterator[int]:
        """Hold one world copy for the block; its index is yielded and given back on exit."""
        with self._lock:
            if not self._free:
                raise RuntimeError(
                    f'no world copy is free: all {self.count} are leased. The run bounds attempts in flight at the slot count, so this is a '
                    'runner bug, not a bench fault.'
                )
            slot = self._free.popleft()
        try:
            yield slot
        finally:
            with self._lock:
                self._free.append(slot)


@dataclass(frozen=True, slots=True)
class _RunConfig:
    """The CLI invocation's TYPED inputs — the raw ``argparse.Namespace`` never leaves ``main()``.

    :meth:`from_args` converts the parsed namespace ONCE into named, typed fields
    (paths are real ``Path``s), so helpers read ``cfg.agent_module_path`` instead of
    dipping into an untyped CLI bag — a typo'd field is a pyright error, not a runtime
    ``AttributeError``. Frozen: everything downstream only reads.
    """

    scenario_set_id: str
    scenario_ids: list[str]
    agent_module_path: Path
    out: Path
    mock_service_url: str | None
    eval_set_id: str | None
    # ONE cap for every scenario in the run. ``None`` until :func:`_resolve_max_turns` resolves it
    # after the set fetch (the ``--max-turns`` flag, else derived from the run's own scenario
    # floors, else the default); every reader past that point reads :attr:`turn_cap`.
    max_turns: int | None
    concurrency: int
    upload_datasets: bool
    # run_name overrides the stamped per-invocation run/experiment handle in BOTH
    # strategies.
    run_name: str | None
    # The PHASE this invocation runs as. It selects DEFAULTS ONLY — the per-turn budget
    # (resolved into ``settings.network`` by ``_settings_from_args``) and whether a stalled
    # turn may spend the re-drive budget. It never overrides a declared value, so an
    # operator who set a budget gets it in both phases.
    run_phase: RunPhase
    # The operator's REQUIRED agent-build tag (``--agent-version``): threaded through
    # Orchestrator.run BOTH onto the create-thread wire (the server stamps it
    # into scenario_metadata) AND into the sink's prepare_run so observability records
    # carry it. Always a non-empty str: the CLI flag overrides, else
    # :func:`_default_agent_version` (``git describe`` / the literal ``"unknown"``).
    agent_version: str
    # ``--trials``: how many independent runs of the set this invocation takes (1 = one run, a single
    # run's shape). REQUIRED: a caller that builds a config has to say how many samples it asked for.
    trials: int
    # ``--rerun-on-turn-cap``: may a scenario that ran OUT of turns with its target still unmet
    # spend ONE automatic re-run at a doubled budget? Default False — a re-run costs a whole
    # scenario's spend, and that is an operator's call, not a default. Declared with a default
    # so a sparse Namespace (unit-test stubs) keeps the default behaviour.
    rerun_on_turn_cap: bool = False
    # Where ``max_turns`` came from, set where it is resolved; ``None`` before that.
    max_turns_source: MaxTurnsSource | None = None
    # ``--egress-probe``: how hard the egress preflight may test each DECLARED production host
    # — a probe CONSENT setting. ``connect`` (default) opens one TCP handshake per host
    # and closes it — no bytes, no TLS, no credentials; ``dns`` resolves the name and sends nothing
    # to the host itself; ``off`` touches the network not at all. Neither ``dns`` nor ``off`` can
    # show a host reachable, so a run that probed in either records ``egress.probe_mode_reduced``.
    egress_probe: ProbeMode = 'connect'
    # ``--scan-contract`` / ``--scan-corpus``: the drift re-check's two inputs — a
    # scan_contract.json a previous `plurai-scan-contract check` wrote, and the corpus roots to
    # re-hash against its recorded inventory. Both absent ⇒ the check reports NOT CHECKED and the
    # run proceeds; a baseline without roots is a check that was ASKED FOR and could not run.
    scan_contract: Path | None = None
    scan_corpus: tuple[Path, ...] = ()
    # ``--scan-suffixes``: the EXTRA file suffixes the recording `plurai-scan-contract check`
    # admitted, used ONLY for a baseline that recorded no scope — a baseline that recorded its
    # ``source_inventory.suffixes`` is re-walked at that scope and this is ignored. For a
    # scope-less baseline a mismatch is not drift and the comparison abstains rather than reporting
    # the out-of-scope files as removed; empty means the default scope.
    scan_suffixes: tuple[str, ...] = ()
    # ``--stability-retry`` / ``--no-stability-retry``: may a FAILED scenario spend one extra
    # run to establish whether its failure reproduces? ``None`` means "take the
    # phase default", resolved by :meth:`stability_retry_enabled`.
    stability_retry: bool | None = None
    # ``--stability-sample``: WHICH second sample a retried failure takes (:data:`SampleKind`).
    # ``None`` means "take the phase default", resolved by :meth:`stability_sample_kind`.
    stability_sample: SampleKind | None = None
    # ``--shape``: the OPERATOR's override of the bench's own ``shape:`` declaration, for
    # debugging only. ``None`` — the ordinary case, and the only one a customer should ever be
    # in — means the harness decides: its ``plurai.config.yaml`` first, dialog otherwise. The AST
    # probe of ``simulate.py`` is a HINT, not a gap-filler: a bench that exports
    # ``invoke_pipeline`` and NO ``invoke_agent`` and declares no ``shape:`` is REFUSED at run
    # start; one that exports BOTH is warned and dispatched as ``dialog``.
    shape: Shape | None = None
    # ``--replace``: may this whole-set run DISCARD a run already measured into ``--out``?
    # Default False — see :func:`_occupied_out_refusal`, the rung that reads it.
    replace: bool = False
    # ``--no-run-pointer``: this run never moves the workspace's ``.last_run`` (:func:`_publish_run_pointer`).
    # A side run into its own --out — regression replays — must not become "the latest run" a later
    # command reads. Default False: an ordinary run publishes the pointer.
    no_run_pointer: bool = False
    # ``--replay-turn-only``: a SINGLE-HOP regression replay — every scenario runs only its issue turn
    # (``Orchestrator.run(replay_turn_only=True)``). Regression folders only; refused beside the flags
    # that re-run a scenario (:func:`_replay_turn_only_refusal`), and its stability retry is off.
    replay_turn_only: bool = False
    # ``--replay-plan``: the run's declaration of how each recorded regression is played (:func:`_load_replay_plan`).
    # ``single_hop_ids`` run only their issue turn, as ``--replay-turn-only`` would, beside every other scenario
    # played whole; ``planned_full_ids`` are the regressions the plan declared whole. ``None`` when no plan was given.
    single_hop_ids: frozenset[str] = frozenset()
    planned_full_ids: frozenset[str] | None = None
    # Set on EVERY trial body's config, trial 1 included, when ``trials >= 2``: which trial it is. ``None``
    # is the invocation itself — the one whose ``--out`` is ``$OUT`` and that owns ``.last_run``.
    trial: _TrialIdentity | None = None
    # The bench declares ``world_state.owner: external`` — or ``owner: agent`` with a ``slots`` count —
    # (:func:`_one_at_a_time_for_world`): ONE shared world that every attempt's ``reset()`` clears and
    # every ``world_snapshot()`` reads, so no two attempts may be in flight at once — ``concurrency`` is
    # 1 and the trials run one after another. With ``world_state.slots: N >= 2`` it is N copies
    # instead: ``concurrency`` is ``min(C, N)``, the trials still run one after another, and every
    # attempt leases one copy from ``world_slots``. Also set, ``concurrency`` untouched, when
    # ``trials`` x ``concurrency`` agent processes do not fit the sandbox's memory limit
    # (:func:`_trials_in_turn_for_memory`): the trials run one after another, each C wide.
    one_at_a_time: bool = False
    world_slots: _WorldSlots | None = None
    # The gathered strategy's exclusive resources instead of that clamp (:func:`_lease_resources`): one lease pool per
    # resource, ``concurrency`` and the trials untouched; empty when the bench declares none.
    resource_leases: tuple[_LeaseSpec, ...] = ()

    @property
    def is_partial(self) -> bool:
        """``--scenario-id`` was given: a NAMED SUBSET of the set, not a fresh population.

        THE definition, because two decisions depend on it and they must never disagree.
        Keeping the prior ``run_summary.json`` at strategy entry and merging it at exit are two
        halves of one policy: clear-but-merge deletes the file the merge needs (and an absent
        prior is the one degrade the merge does not log, so the total shrinks in silence), while
        keep-but-replace overwrites the cumulative file with this run's subset — the very defect
        the merge exists to prevent. Derived once, so a mismatch is unrepresentable rather than
        merely untested.
        """
        return bool(self.scenario_ids)

    def single_hop(self, scenario_id: str) -> bool:
        """Does ``scenario_id`` run only its issue turn: the whole run is ``--replay-turn-only``, or the plan says so."""
        return self.replay_turn_only or scenario_id in self.single_hop_ids

    @property
    def stability_retry_enabled(self) -> bool:
        """Does this run spend a second sample on ordinary failures?

        OFF on a single-hop replay (``--replay-turn-only``): the K replays ARE its samples.

        Default ON in BOTH phases; ``--no-stability-retry`` turns it off. The
        phases differ in the KIND of second sample (:meth:`stability_sample_kind`): an agent
        re-run costs a whole scenario's spend, while a full run's default sample is a judge
        replay, which spends judge tokens only.

        The INVALID-run retry is NOT gated on this: that attempt
        measured nothing, so re-running it is the first sample rather than a second, and it
        recovers spend rather than doubling it. See :meth:`_StabilityBudget.claim`.
        """
        if self.replay_turn_only:
            return False
        return True if self.stability_retry is None else self.stability_retry

    @property
    def stability_sample_kind(self) -> SampleKind:
        """WHICH second sample a retried failure takes: the operator's ``--stability-sample``, else
        the phase's own (:data:`_PHASE_SAMPLE_KIND`). One table, so the phase that decides a
        redraw and the phase that annotates a verdict cannot swap questions by accident."""
        return self.stability_sample if self.stability_sample is not None else _PHASE_SAMPLE_KIND[self.run_phase]

    @property
    def is_scored(self) -> bool:
        """An eval set was attached — a PRECONDITION for grading, never evidence of it.

        Read the name narrowly. This says a ``--eval-set-id`` was passed, nothing more: the set
        may resolve to zero policies, every judge may error, and a finalize may never complete.
        It must never be written into ``run_summary.json`` as ``scored`` — a key documented to
        mean "every row here was graded" — because that passes a statement about a command line
        off as a statement about a measurement. Grading claims come from
        :func:`_scenario_was_graded`; this only says whether one is even possible.

        Truthiness, matching every other read of this field in the module — ``--eval-set-id ''``
        must behave the same way everywhere.
        """
        return bool(self.eval_set_id)

    @property
    def turn_cap(self) -> int:
        """The resolved turn cap. Reading it before :func:`_resolve_max_turns` ran is a wiring defect."""
        if self.max_turns is None:
            raise RuntimeError('max_turns was read before the run resolved it (_resolve_max_turns)')
        return self.max_turns

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, mock_service_url: str | None = None) -> _RunConfig:
        """The ONE Namespace→typed-fields conversion (argparse stays in main()).

        ``mock_service_url`` is the RESOLVED url (flag > env > ``.mock_service_url`` sidecar),
        threaded in from ``_settings_from_args`` rather than re-read off the namespace: the
        sidecar rung exists only after that resolution, and reading the raw flag here would
        make the config disagree with the url the bearer guard and the health probe checked.
        Defaults to the flag for a caller that has not resolved settings.
        """
        single_hop_ids: frozenset[str] = frozenset()
        planned_full_ids: frozenset[str] | None = None
        if raw_plan := getattr(args, 'replay_plan', None):
            single_hop_ids, planned_full_ids, notes = _load_replay_plan(Path(raw_plan))
            for note in notes:
                print(f'[regressions] WARNING: {note}', file=sys.stderr)
        return cls(
            scenario_set_id=args.scenario_set_id,
            # Empty ⇒ run the whole set (resolved via _resolve_run_scope's server fetch).
            scenario_ids=list(args.scenario_id or []),
            agent_module_path=Path(args.agent_module_path),
            agent_version=args.agent_version or _default_agent_version(Path(args.agent_module_path).parent),
            trials=getattr(args, 'trials', 1),
            out=Path(args.out),
            mock_service_url=mock_service_url if mock_service_url is not None else args.mock_service_url,
            eval_set_id=args.eval_set_id,
            max_turns=args.max_turns,
            max_turns_source='flag' if args.max_turns is not None else None,
            concurrency=args.concurrency,
            upload_datasets=args.upload_datasets,
            run_name=args.run_name,
            # getattr: a sparse Namespace (unit-test stubs that omit the newer flags) falls back
            # to the full-run phase, the same contract _settings_from_args documents.
            run_phase=getattr(args, 'run_phase', 'full'),
            rerun_on_turn_cap=bool(getattr(args, 'rerun_on_turn_cap', False)),
            egress_probe=getattr(args, 'egress_probe', 'connect'),
            scan_contract=(Path(raw) if (raw := getattr(args, 'scan_contract', None)) else None),
            scan_corpus=tuple(Path(root) for root in (getattr(args, 'scan_corpus', None) or [])),
            scan_suffixes=tuple(str(s) for s in (getattr(args, 'scan_suffixes', None) or [])),
            stability_retry=getattr(args, 'stability_retry', None),
            stability_sample=getattr(args, 'stability_sample', None),
            shape=getattr(args, 'shape', None),
            replace=bool(getattr(args, 'replace', False)),
            no_run_pointer=bool(getattr(args, 'no_run_pointer', False)),
            replay_turn_only=bool(getattr(args, 'replay_turn_only', False)),
            single_hop_ids=single_hop_ids,
            planned_full_ids=planned_full_ids,
        )


#: The ``--replay-plan`` file's schema (the plugin's ``regressions.py replay-plan --out``). A plan without a schema
#: key is read the same way; one naming another schema is not read.
REPLAY_PLAN_SCHEMA: Final = 'replay-plan-v1'


def _load_replay_plan(path: Path) -> tuple[frozenset[str], frozenset[str] | None, list[str]]:
    """``(single_hop ids, full ids, warnings)`` off a ``--replay-plan`` file: ``{schema?, single_hop: [...], full: [...]}``.

    Never refuses: an unreadable plan is no plan (``None`` full ids — every regression is played whole and said so), an
    id without the regression prefix in ``single_hop`` is played whole (only a recorded regression has a cut), and an id
    in both lists is played whole (the plan's ``full`` is the conservative reading). Each case is one warning line.
    """
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return frozenset(), None, [f'the replay plan {path} could not be read ({type(exc).__name__}: {exc}); every recorded regression is played whole.']
    if not isinstance(doc, dict) or doc.get('schema', REPLAY_PLAN_SCHEMA) != REPLAY_PLAN_SCHEMA:
        return frozenset(), None, [f'the replay plan {path} is not a {REPLAY_PLAN_SCHEMA} object; every recorded regression is played whole.']
    notes: list[str] = []

    def ids(key: str) -> set[str]:
        raw = doc.get(key)
        if raw is None:
            return set()
        if not isinstance(raw, list):
            notes.append(f'the replay plan {path} has a `{key}` that is not a list; it is ignored.')
            return set()
        return {str(item) for item in raw if isinstance(item, str) and item}

    single, full = ids('single_hop'), ids('full')
    if not_regressions := sorted(sid for sid in single if not is_regression_scenario(sid)):
        notes.append(f'the replay plan names {not_regressions} single-hop, but only a recorded regression has a cut; they are played whole.')
    if both := sorted(single & full):
        notes.append(f'the replay plan names {both} both single-hop and full; they are played whole.')
    single = {sid for sid in single if is_regression_scenario(sid)} - full
    return frozenset(single), frozenset(full), notes


def _replay_plan_gaps(cfg: _RunConfig, scenario_ids: Sequence[str]) -> str | None:
    """The one warning line for the recorded regressions this run plays whole WITHOUT a declaration saying so, else ``None``.

    A plan-routed full folder is a declaration, not a gap; a ``--replay-turn-only`` run plays every one at its cut.
    """
    if cfg.replay_turn_only:
        return None
    declared = cfg.single_hop_ids | (cfg.planned_full_ids or frozenset())
    undeclared = [sid for sid in scenario_ids if is_regression_scenario(sid) and sid not in declared]
    if not undeclared:
        return None
    if cfg.planned_full_ids is None:
        return f'{len(undeclared)} recorded regression(s) played as whole conversations (no replay plan): {undeclared}'
    return f'{len(undeclared)} recorded regression(s) are not in the replay plan and are played as whole conversations: {undeclared}'


def _build_orchestrator(cfg: _RunConfig, settings: Settings, sink: ObservabilitySink) -> Orchestrator:
    """The ONE Orchestrator construction both run strategies share.

    Orchestrators are short-lived (one per scenario / one per setup step); the sink is
    shared across all of them and is NEVER closed by an orchestrator — each strategy
    owns the single ``sink.close()`` at its end.
    """
    return Orchestrator(
        simulate_path=cfg.agent_module_path,
        settings=settings,
        observability=sink,
        client=build_default_client(settings),
        shape=cfg.shape,
    )


#: The ledger ``outcome`` a row carries when its ceiling re-run failed and the attempt it
#: superseded was restored as the row itself (see :func:`_restore_superseded_attempt`).
_RERUN_RESTORED_OUTCOME = 'restored_superseded_attempt'


def _restored_rerun_count(turn_cap_reruns: dict[str, Any]) -> int:
    """How many ceiling re-runs FAILED and gave their superseded attempt back.

    These rows are the ones with no ``previous_attempt/`` on disk — the restore removes it —
    which is why the summary line cannot make one blanket claim about where to look.
    """
    return sum(1 for block in turn_cap_reruns.values() if isinstance(block, dict) and block.get('outcome') == _RERUN_RESTORED_OUTCOME)


#: The remedy each ``ended_early`` reason routes to. They are DIFFERENT OWNERS — a budget the
#: operator sets, a defect in ``simulate.py``, a finding about the scenario — which is exactly
#: why the summary line is built from the typed reasons rather than from a count.
_ENDED_EARLY_REMEDIES: dict[str, tuple[str, str]] = {
    TURN_CAP_REASON: ('operator', 'the turn budget ran out mid-conversation — raise --max-turns'),
    # A turn that failed after the dialog started LANDS (graded to the turn it reached) instead of
    # discarding the turns before it; each names the owner of the failure.
    AGENT_RAISED_REASON: (
        'agent',
        "the agent's own code raised on its turn and produced no reply — the scenario LANDED and was graded to the turn "
        'it reached. Read turn_failure in the transcript for the exception',
    ),
    AGENT_RECURSION_REASON: (
        'agent',
        "the agent's graph hit its recursion/step limit — the scenario LANDED and was graded to the turn it reached. "
        'turn_failure.pre_crash_tool_activity says whether its tools were answering before the loop (an empty mock can loop a graph too)',
    ),
    PROVIDER_INFRA_REASON: (
        'operator',
        "the agent's model provider refused or failed the call — infrastructure, not the agent. The scenario LANDED to the "
        'turn it reached (recorded `incomplete`, the trial invalid); check the provider quota and status, then re-run',
    ),
    TURN_BUDGET_REASON: (
        'operator',
        'one agent turn ran past the per-turn wall — the scenario LANDED to the turn it reached (recorded `incomplete`, '
        'the trial invalid). Raise --agent-turn-timeout if a turn of this agent is legitimately that slow',
    ),
    TIME_BUDGET_REASON: (
        'operator',
        'the per-scenario time budget ran out mid-conversation — the scenario LANDED and was graded to the turn it '
        'reached (recorded `incomplete`, neither passed nor failed). Raise --scenario-timeout if this flow is legitimately '
        'this long, or find out why the turns are slow',
    ),
    # A give-up has NO route of its own: it says who stopped, and each row routes by the CAUSE the
    # simulator stated (or by a voiding breach) — the text is only the heading above those rows.
    USER_GAVE_UP_REASON: (
        'unknown',
        'the simulated user declared its goal unreachable and stopped trying — each row routes by a fault that voided its '
        'scenario, else by the cause it stated:',
    ),
    # THREE owners look like this from here: the agent said
    # nothing, `invoke_agent` returns its reply where the SDK does not read it, or the SDK could not
    # read a shape `invoke_agent` DOES return — so the route names all three, never simulate.py alone.
    ADAPTER_DEGRADED_REASON: (
        'harness',
        'EVERY turn was emptied by the harness, so nothing about the agent was measured — the agent said nothing, '
        '`invoke_agent` returns its reply where the SDK does not read it, or the SDK could not read a shape `invoke_agent` '
        'DOES return (`surface: sdk` — file it, do not edit simulate.py around it)',
    ),
    # A HARNESS finding, and the one whose door is neither simulate.py nor the agent: the
    # SERVER's user simulator is what produced nothing. Routed to the draw — the scenario's
    # persona and its language — because that is what the simulator was given and everything the
    # operator can act on is there. Nothing in this run is a verdict about the agent, so the line
    # says so before it says anything else; the unmet obligations below it were never asked for.
    SIM_DEGENERATE_REASON: (
        'backend',
        'the simulated user produced no turn — its draw and its one re-draw both came back unusable, so the dialog '
        'stopped and nothing about the agent was measured. A backend fault: re-run the scenario, and if it '
        "repeats, look at the scenario's persona and declared language",
    ),
    # The two AGENT-side endings. Neither is a defect on its own — a kiosk, an IVR and a support
    # flow all legitimately close after the transaction, and a handoff is a designed exit — so
    # the remedy routes to the SCENARIO, which is the only place that says whether the agent was
    # supposed to be still talking. Obligations the dialog never reached still score as unmet,
    # and that reading is honest exactly when the scenario expected more.
    AGENT_ENDED_REASON: (
        'scenario',
        'the AGENT ended the conversation itself — check the scenario expected it to, before reading the unmet obligations as failures',
    ),
    HANDED_OFF_REASON: ('scenario', 'the AGENT transferred the conversation to another party — the measurement stops where the handoff did'),
    # The two BACKEND endings, routed by owner like the rest: a schedule the SCENARIO set, and a
    # verdict an EVAL reached. Neither is a finding about the agent on its own.
    PLAN_EXHAUSTED_REASON: (
        'scenario',
        "the SCENARIO's schedule was fully delivered — every premise beat played and every planted question asked — so the "
        'server stopped instead of padding to the cap; not a budget, so do not raise --max-turns. If a policy went unmet, check '
        'whether the dialog ever asked for it: an obligation nobody requested is a scenario-coverage finding, not an agent failure',
    ),
    POLICY_TERMINATED_REASON: ('eval', "an EVAL stopped the run — one of its policies declared it unsalvageable; read that policy's verdict first"),
    # Not driven: the seed did not read back at Init (``plurai.world.readback``). The voiding breach the backend records
    # on the same pair (``undelivered_seed``) routes the row first; this is the line for a summary without it.
    SEED_NOT_LANDED_REASON: (
        'harness',
        "the scenario's seed did not read back as seeded at Init, so it was not driven and is unmeasured (undelivered_seed) — "
        "make the harness's seed_state apply the named rows and its world_snapshot return them; the agent was not measured",
    ),
    # The run's own choice, not a finding: a single-hop regression replay stops after the turn under test.
    REPLAY_CUT_REASON: (
        'operator',
        'a single-hop regression replay (--replay-turn-only or the --replay-plan) stops by design after the turn under test; its end-of-run '
        'checks were graded on that one turn, and a check that needed later turns says so instead of scoring',
    ),
}

#: A ``plan_exhausted`` row whose last simulated-user message PUSHED BACK: the schedule was spent,
#: and the run ended on a dispute about the agent's reply — the agent's claim is the finding.
_PLAN_EXHAUSTED_PUSHBACK_REMEDY = (
    "the schedule was spent and the simulated user's last message disputed the agent's reply — read that agent turn and its answer to the push-back"
)

#: An agent-side close (``agent_ended`` / ``handed_off``) that came BEFORE the scenario's target beat
#: (the backend's ``dialog_outcome.preempted_target``): the target never arose because the agent ended the
#: dialog, so the goal failed on ``agent_closed`` and the finding is the agent's closing turn.
_AGENT_CLOSED_BEFORE_TARGET_REMEDY = (
    'the AGENT closed the conversation before the scenario target beat {beat} arose — read that agent turn: did the simulated user ask for it?'
)

#: The remedy each stated give-up CAUSE routes to (``orchestrator.GIVE_UP_CAUSE_*``). Different
#: owners again: three are about the agent (and the scenario that expected otherwise), and one
#: is about the harness.
_GIVE_UP_CAUSE_REMEDIES: dict[str, tuple[str, str]] = {
    GIVE_UP_CAUSE_AGENT_REFUSED: ('agent', 'the agent REFUSED the goal — look at what it refused, then at whether the scenario expected it to'),
    GIVE_UP_CAUSE_AGENT_CANNOT: ('agent', 'the agent said it CANNOT do it — look at the agent (its tools, its instructions), then at the scenario premise'),
    GIVE_UP_CAUSE_AGENT_REPEATED: ('agent', 'the agent gave the same non-answer more than once — look at the agent'),
    GIVE_UP_CAUSE_AGENT_DIVERGED: (
        'agent',
        'the agent kept answering something OTHER than what was asked and did not come back when pushed — look at the agent',
    ),
    GIVE_UP_CAUSE_REPLY_NOT_VISIBLE: (
        'harness',
        "the simulated user could not see the agent's reply (blank, or produced where the user is never shown it) — a HARNESS "
        "finding: fix simulate.py so every reply the agent produces reaches the user, before reading this as the agent's or the scenario's",
    ),
}

#: The bucket a give-up lands in when no cause is ON RECORD — the backend stated none, or the
#: row was carried from a summary written before this SDK recorded causes
#: (``ended_early_causes_unrecorded``), and the line does not claim to know which. It asserts no
#: remedy either: there is nothing measured to route on.
_NO_CAUSE_ON_RECORD = 'no cause on record'
_NO_CAUSE_REMEDY = (
    'the backend stated none, or the row predates this SDK recording causes (see ended_early_causes_unrecorded) — read the transcript '
    'before routing this to the agent or the scenario'
)


def _resolve_ending(reason: str, *, cause: str | None, disputed_turn: int | None, breaches: Iterable[Any], preempted_target: str | None) -> dict[str, str]:
    """THE ONE ROUTE for a row that ended early: ``{owner, remedy}``, plus ``qualifier`` when the
    route is narrower than the reason — the only reader of the remedy tables. The console prints
    these and ``run_summary.json`` publishes them (``ended_early_routes``) for the run report.

    Precedence, each keyed on a typed fact and never on a channel name:

    1. a breach that VOIDS the scenario (``voids_scenario`` on the backend's own record) owns the
       ending — its owner, and its own detail as the one remedy. The agent ran a different
       scenario, so whatever the simulated user blamed, the fault is the breach owner's;
    2. a ``plan_exhausted`` row that ended on a push-back routes to the disputed agent turn; an
       agent-side close that came before the scenario's target beat (``preempted_target``) routes
       to the agent;
    3. a stated cause routes by the cause; a give-up with none asserts no route;
    4. otherwise the reason's own route.
    """
    voiding = [b for b in breaches if isinstance(b, Mapping) and b.get('voids_scenario') is True and isinstance(b.get('detail'), str)]
    if voiding:
        owner = str(voiding[0].get('owner') or 'harness').split('@', 1)[0]
        return {'owner': owner, 'qualifier': f'voided by a {owner} fault', 'remedy': ' '.join(b['detail'] for b in voiding)}
    if reason == PLAN_EXHAUSTED_REASON and disputed_turn is not None:
        return {'owner': 'agent', 'qualifier': 'after a push-back', 'remedy': f'{_PLAN_EXHAUSTED_PUSHBACK_REMEDY} — agent turn {disputed_turn} (0-based)'}
    if reason in (AGENT_ENDED_REASON, HANDED_OFF_REASON) and preempted_target:
        return {'owner': 'agent', 'qualifier': 'before the target', 'remedy': _AGENT_CLOSED_BEFORE_TARGET_REMEDY.format(beat=preempted_target)}
    if cause:
        owner, remedy = _GIVE_UP_CAUSE_REMEDIES.get(cause, ('unknown', 'cause not known to this SDK build — carried verbatim; read the transcript'))
        return {'owner': owner, 'qualifier': cause, 'remedy': remedy}
    if reason == USER_GAVE_UP_REASON:
        return {'owner': 'unknown', 'qualifier': _NO_CAUSE_ON_RECORD, 'remedy': _NO_CAUSE_REMEDY}
    owner, remedy = _reason_route(reason)
    return {'owner': owner, 'remedy': remedy}


def _reason_route(reason: str) -> tuple[str, str]:
    """The reason's own ``(owner, remedy)`` — for a give-up, the heading above its routed rows."""
    return _ENDED_EARLY_REMEDIES.get(reason, ('unknown', 'reason not known to this SDK build — see the run summary and the transcripts'))


def _ended_early_lines(reasons: Mapping[str, str], routes: Mapping[str, Mapping[str, str]], unrecorded: Sequence[str] = ()) -> list[str]:
    """One line per DISTINCT reason with its route (``routes``, from :func:`_resolve_ending`), and a
    line per narrower route beneath or beside it.

    A line built from a bare count would have to list every remedy at once — "raise --max-turns,
    fix simulate.py, or look at the scenario" — for whichever one actually happened. A run whose
    scenarios ALL ended because the simulated user gave up would be told to raise its turn
    budget; the reasons are typed end to end, so the line routes by reason.

    A give-up's heading says only who stopped, and every give-up row lands in a sub-line under it:
    its stated cause, ``no cause on record`` (which asserts no route, and says how many rows were
    carried from a summary that never recorded causes), or a voiding breach. Any other reason's
    unqualified rows share its heading; a row with a narrower route (a push-back, a voiding breach)
    gets its own line naming the rows.

    A reason or cause this build does not recognise still gets a line: both vocabularies are
    shared with the backend on an independent release cycle, and an unnamed value is still more
    than a count.
    """
    by_reason: dict[str, list[str]] = {}
    for sid, reason in reasons.items():
        by_reason.setdefault(reason, []).append(sid)
    lines = []
    for reason, sids in sorted(by_reason.items(), key=lambda item: (-len(item[1]), item[0])):
        narrow: dict[tuple[str, str], list[str]] = {}
        plain = []
        for sid in sorted(sids):
            route = routes.get(sid) or _resolve_ending(reason, cause=None, disputed_turn=None, breaches=(), preempted_target=None)
            if route.get('qualifier'):
                narrow.setdefault((route['qualifier'], route['remedy']), []).append(sid)
            else:
                plain.append(sid)
        give_up = reason == USER_GAVE_UP_REASON
        if give_up or plain:
            lines.append(f'  - {len(sids) if give_up else len(plain)} x {reason}: {_reason_route(reason)[1]}')
        for (qualifier, remedy), named in sorted(narrow.items(), key=lambda item: (-len(item[1]), item[0])):
            if give_up:
                never_asked = sum(1 for sid in named if sid in set(unrecorded)) if qualifier == _NO_CAUSE_ON_RECORD else 0
                label = f'{qualifier} ({never_asked} recorded before this SDK carried causes)' if never_asked else qualifier
                lines.append(f'      · {len(named)} x {label}: {remedy}')
            else:
                lines.append(f'  - {len(named)} x {reason} ({qualifier}): {remedy} [{", ".join(named)}]')
    return lines


def _print_summary(
    *,
    total: int,
    failures: int,
    incomplete: int = 0,
    succeeded: int | None = None,
    dataset_url: str | None = None,
    run_links: RunLinks | None = None,
    scored: bool = True,
    exit_code_covers: str = 'all',
    ran_count: int | None = None,
    graded_count: int | None = None,
    eval_set_attached: bool | None = None,
    ended_early: Mapping[str, str] | None = None,
    ended_early_causes: Mapping[str, str] | None = None,
    ended_early_causes_unrecorded: Sequence[str] = (),
    degraded_count: int = 0,
    degraded_turns: int = 0,
    untrustworthy: Mapping[str, str] | None = None,
    echoes_dropped: Mapping[str, Mapping[str, int]] | None = None,
    unclean_user_sim: Mapping[str, Any] | None = None,
    unchecked_user_sim: Mapping[str, Any] | None = None,
    noted_user_sim: Mapping[str, Any] | None = None,
    turn_cap_rerun_count: int = 0,
    turn_cap_rerun_restored: int = 0,
    world_unread: Mapping[str, Mapping[str, Any]] | None = None,
    host_toolchain_stamp: str | None = None,
    scenario_verdicts: Mapping[str, str] | None = None,
    scenario_high_severity: Mapping[str, Sequence[str]] | None = None,
    ended_early_routes: Mapping[str, Mapping[str, str]] | None = None,
    measurement_clean: bool | None = None,
    measurement_unclean_because: Sequence[str] = (),
) -> None:
    """The operator-facing counterpart of ``run_summary.json`` — the SAME resolved numbers.

    Both strategies feed this from the ``_RunVerdict`` the writer returns, because deriving
    the two separately lets them disagree on screen: a ``--scenario-id`` retry prints
    ``1/1 scenarios succeeded`` while the file it writes beside it says ``total: 7``, and the
    merged ``scored`` floor — the one signal saying the older rows were never graded — lives
    and dies inside the writer's frame. ``exit_code_covers`` is the same scope tag the file
    carries; on a merged run the counts span attempts, so the line says which population it
    just described rather than leaving the reader to assume it was this one.
    """
    # The COUNTED value, for the same reason run_summary.json carries one: a residual reports any
    # row status it does not recognise as a success, and the screen and the file would then
    # disagree in exactly the case the verdict warns about. Falls back for the direct callers
    # (tests, early-exit paths) that have no verdict to hand.
    # ROW STATUS says the thread ran to completion (and drives the exit code); the VERDICT
    # tally beside it is the house verdict per row, so "3/3" can never read as three passes.
    headline = f'\n{total - failures - incomplete if succeeded is None else succeeded}/{total} scenarios ran to completion'
    if scenario_verdicts is not None:
        tally = dict(Counter(scenario_verdicts.values()))
        # A high-severity red fails its row, so a passed row carries none. A passed row whose
        # backend served no list is UNKNOWN on that axis, never a zero.
        passed_ids = [sid for sid, status in scenario_verdicts.items() if status == 'passed']
        unreported = sum(1 for sid in passed_ids if sid not in scenario_high_severity) if scenario_high_severity is not None else 0
        parts = [
            f'{tally.pop("passed", 0)} passed' + (f' (high-severity not reported for {unreported})' if unreported else ''),
            f'{tally.pop("failed", 0)} failed',
            *(f'{n} {k.replace("_", " ")}' for k, n in sorted(tally.items())),
        ]
        if len(scenario_verdicts) < total:
            parts.append(f'{total - len(scenario_verdicts)} without a verdict')
        headline += f' — verdicts: {", ".join(parts)}'
    if measurement_clean is False:
        headline += f' · measurement NOT clean ({", ".join(measurement_unclean_because) or "unnamed"})'
    print(f'{headline}.')
    # The ONE stamp a run the operator let start over a host gap carries: every row was graded, and
    # the rows the gap touched graded the agent's fallback path. Beside the headline, never a count.
    if host_toolchain_stamp:
        print(f'HOST TOOLCHAIN: {host_toolchain_stamp}. See host_toolchain in {RUN_SUMMARY_FILENAME}.')
    if incomplete:
        print(
            f'INCOMPLETE: {incomplete} scenario(s) took no full measurement. A row that ran out of its per-scenario '
            f'time budget LANDED — its transcript, tool events and world state are captured to the turn it reached, '
            f'and the judges graded what they could see. A row the SUITE quarantined never ran: its own reference '
            f'path cannot pass its eval set (error.json kind {ScenarioQuarantined.kind}), so it carries no artifacts '
            f'and no grading. Neither shape passed nor failed, and both are counted apart; see incomplete + '
            f'ended_early_scenarios (and solvability.quarantined_scenarios) in {RUN_SUMMARY_FILENAME}.'
        )
    # Only when the merge actually brought rows in. A first ``--scenario-id`` run into a fresh
    # --out is scoped as ``ran_this_invocation`` too, but its counts describe exactly what it
    # just ran — calling that "cumulative across attempts" would be its own small untruth.
    if exit_code_covers == 'ran_this_invocation' and (ran_count is None or total > ran_count):
        print('(cumulative across attempts in this --out; the exit code covers only the scenarios THIS invocation ran.)')
    # "Ran to completion" means RAN, not PASSED. Every ungraded per-scenario line prints
    # pass_rate=0.000 aggregate_score=0.000 status=completed, which is byte-identical to a run
    # that failed every policy, and the exit code is 0 either way. Without these lines the only
    # trace of "nothing was measured" is four nulls buried in each transcript's eval block.
    # Printed last so they are the final thing on screen.
    #
    # THREE cases, because collapsing them makes the banner misleading: a single line names
    # the only cause it knows ("no --eval-set-id was passed") and therefore stays silent for
    # a run that DID pass one and still measured nothing.
    if not scored:
        if not eval_set_attached:
            print('NOT SCORED: no --eval-set-id was passed, so no policy was evaluated. Every 0.000 above means "not measured", not "failed".')
        elif graded_count:
            print(
                f'PARTIALLY SCORED: only {graded_count} of {total} scenario(s) in this summary produced a policy '
                f'verdict. For the rest, every 0.000 means "not measured", not "failed" — see graded_scenarios in '
                f'{RUN_SUMMARY_FILENAME}.'
            )
        else:
            print(
                'NOT SCORED: an eval set was attached but NOT ONE policy verdict landed — every scenario either '
                'failed to finalize or finalized over policies that errored or never applied. Every 0.000 above '
                'means "not measured", not "failed".'
            )
    # A scenario the HARNESS stopped is not a scenario the agent finished, and on screen the
    # two are the same line: it counts as succeeded and its evals are low for obligations the
    # conversation never reached. Broken out BY REASON, because the remedies have different
    # owners and a lumped line sends every operator down the same one.
    if ended_early:
        print(
            f'ENDED EARLY: {len(ended_early)} scenario(s) ended for something other than the user getting what they '
            f'came for. Anything their dialog never reached scores as unmet — by reason:'
        )
        # A direct caller hands no routes; they resolve through the same resolver from what it did hand.
        routes = (
            ended_early_routes
            if ended_early_routes is not None
            else {
                sid: _resolve_ending(reason, cause=(ended_early_causes or {}).get(sid), disputed_turn=None, breaches=(), preempted_target=None)
                for sid, reason in ended_early.items()
            }
        )
        for line in _ended_early_lines(ended_early, routes, ended_early_causes_unrecorded):
            print(line)
        print(f'  See ended_early_scenarios (and ended_early_causes) in {RUN_SUMMARY_FILENAME} for the reason per row.')
    # The loudest line on this screen: these turns carry no reply the runner could read, so every
    # judge that scored them may not have scored the agent. It names NO owner — no typed fact says
    # who emptied the turn — and defers to the per-turn warning, which routes by subkind.
    if degraded_count:
        print(
            f'EMPTY AGENT TURNS: {degraded_turns} turn(s) across {degraded_count} scenario(s) came back with no reply '
            f'the runner could read and were scored as silence. Before acting on a verdict over one, check the owners '
            f'in the order its `scoring an EMPTY agent turn` warning gives (the agent, simulate.py, the runner). See '
            f'degraded_scenarios in {RUN_SUMMARY_FILENAME}; a scenario whose EVERY turn came back empty is also listed '
            f'in ended_early_scenarios as adapter_degraded.'
        )
    # Its sibling breach, and the same shape: a run fully "succeeded" and "scored" while the tools
    # the agent called answered for a world they never read. ROUTES rather than blames — the agent's
    # call is the one that came back empty, and a seeded record that truly owns nothing of what was
    # asked is the one honest reading the runner cannot rule out.
    if world_unread:
        tools = sorted({str(tool) for row in world_unread.values() for tool in row.get('tools') or []})
        print(
            f'WORLD UNREAD: {sum(int(row.get("calls") or 0) for row in world_unread.values())} tool call(s) across '
            f'{len(world_unread)} scenario(s) asked only for values ONE seeded record carries and came back with zero rows '
            f'({", ".join(tools)}) — the mock env (or the seed_state in simulate.py) most likely never read the world the scenario '
            f'was seeded with, so verdicts resting on those reads grade the environment, not the agent (unless that record '
            f'genuinely owns none of what was asked). See world_unread_scenarios in {RUN_SUMMARY_FILENAME}.'
        )
    # The row the backend SCORED over a world it does not vouch for. Its own banner, beside the one
    # above, because the two conditions are the same shape: a run that is fully "succeeded" and
    # fully "scored" while some of what was measured is not what the number says it is.
    #
    # It needs one because the row keeps its cells (they landed), so `scored` is True and the
    # `PARTIALLY SCORED` block above does not fire for it. The only other trace is one mid-run
    # line on STDERR from `_record_untrustworthy`, while this function writes STDOUT, so
    # `> run.log` would keep `7/7 scenarios succeeded.` and lose the caveat entirely.
    if untrustworthy:
        print(
            f'UNTRUSTWORTHY WORLD: {len(untrustworthy)} scenario(s) were SCORED over a final world the backend does not '
            f'vouch for ({"; ".join(f"{sid}: {reason}" for sid, reason in sorted(untrustworthy.items()))}). Their cells are '
            f'kept — they landed — but a check that reads the world resolves against nothing there, so a PASS on those rows '
            f'may be vacuous. See untrustworthy_scenarios in {RUN_SUMMARY_FILENAME}.'
        )
    # The other client-side-only fact about that same channel, and ONLY for the rule that is a
    # defect. A declared user role is the ordinary LangGraph shape and is recorded in the file
    # without a line here — announcing it would put a "your adapter is broken" banner on every
    # correct LangGraph run.
    if breached := {sid: counts.get('verbatim_echo', 0) for sid, counts in (echoes_dropped or {}).items() if counts.get('verbatim_echo')}:
        print(
            f'ROLE-LESS USER ECHOES: {sum(breached.values())} message(s) across {len(breached)} scenario(s) came back '
            f'inside the AGENT chain carrying no role and the words this harness had just sent, verbatim. They were '
            f'kept OUT of the agent channel (so nothing scored them as the agent), but simulate.py should return only '
            f'what the AGENT produced, with a role on every message. See echoes_dropped_scenarios in {RUN_SUMMARY_FILENAME}.'
        )
    # The OTHER side of the apparatus: the simulated USER departing from the scenario it was
    # given. Its own line, not folded into the one above, because the remedy is different — no
    # eval change repairs a transcript that is unfaithful to the scenario.
    if unclean_user_sim:
        kinds = sorted({kind for block in unclean_user_sim.values() for kind in _user_sim_kinds(block)})
        print(
            f'USER-SIM FIDELITY: {len(unclean_user_sim)} scenario(s) carry a dialog the simulated USER did not drive '
            f'as the scenario specified{" (" + ", ".join(kinds) + ")" if kinds else ""} — so what was scored is not '
            f'quite the scenario that was authored. This is a SCENARIO/simulator finding, not an eval one: no eval '
            f'change repairs it. See user_sim_integrity in {RUN_SUMMARY_FILENAME}.'
        )
    # NOT the line above: nothing departed and nothing was cleared. The backend's evaluator-voice
    # judge did not answer for those turns (``judge_unavailable``), so the run's verdict is UNKNOWN
    # on this channel rather than unclean. Said separately so a judge outage does not read as a
    # simulator defect.
    if unchecked_user_sim:
        turns = sum(len(block.get('evaluator_voice_unchecked') or {}) for block in unchecked_user_sim.values() if isinstance(block, Mapping))
        print(
            f'USER-SIM FIDELITY NOT MEASURED: {len(unchecked_user_sim)} scenario(s) carry {turns} user turn(s) the evaluator-voice '
            f'judge did not answer for (judge_unavailable), so whether the simulator stayed in persona there was not decided — '
            f'nothing departed, nothing was cleared. Re-run those scenarios to measure them (a re-score reads the recorded verdicts). '
            f'See user_sim_integrity in {RUN_SUMMARY_FILENAME}.'
        )
    # The simulator's own delivery LEDGER disagreeing with itself (a beat counted past what the
    # premise declares) — the backend folds it BESIDE the departures, and it leaves ``clean`` alone
    # because the words stayed on schedule. Its own line so the turns get read; folded into the
    # line above it would mark a run unclean over one miscounted beat.
    if noted_user_sim:
        unreadable = sorted(sid for sid, block in noted_user_sim.items() if not isinstance(block.get('self_report'), dict))
        if unreadable:
            print(f'USER-SIM LEDGER: the self-report table on {", ".join(unreadable)} is not a mapping — unreadable, neither cleared nor flagged.')
        kinds = sorted({kind for block in noted_user_sim.values() for kind in _user_sim_kinds(block, 'self_report')})
        print(
            f'USER-SIM LEDGER: {len(noted_user_sim)} scenario(s) carry a self-report the simulator could not reconcile'
            f'{" (" + ", ".join(kinds) + ")" if kinds else ""} — the transcript stayed on schedule and no verdict is discounted; '
            f'read those turns before acting on one. See user_sim_integrity.self_report in {RUN_SUMMARY_FILENAME}.'
        )
    # A re-run at a different turn budget is not the same measurement as the attempt it
    # replaced, and both are on disk — say so rather than letting the second one pass for the
    # only one that ever ran.
    if turn_cap_rerun_count:
        # WHERE the superseded attempt is depends on how the re-run went, so the line says which
        # case each row is rather than making one blanket claim: a re-run that FAILED restores
        # the earlier attempt to the top level and removes previous_attempt/ entirely, so
        # telling every operator to look there would send some of them to a directory not on disk.
        kept = turn_cap_rerun_count - turn_cap_rerun_restored
        where = []
        if kept:
            where.append(f'{kept} kept the superseded attempt under {PREVIOUS_ATTEMPT_DIRNAME}/ in its scenario dir')
        if turn_cap_rerun_restored:
            where.append(
                f'{turn_cap_rerun_restored} had the re-run FAIL, so the earlier attempt was restored as the row '
                f'itself (see turn_cap_rerun_failed in its transcript)'
            )
        print(
            f'TURN-CAP RE-RUNS: {turn_cap_rerun_count} scenario(s) ran out of turns mid-goal and were re-run once at '
            f'a raised budget — {"; ".join(where)}. See turn_cap_reruns in {RUN_SUMMARY_FILENAME} for both budgets.'
        )
    # The platform link to the recorded experiment/dataset — the "where do I look"
    # line. Optional: None (no sink / unresolvable) prints nothing. This surface
    # reaches customers: counts + link ONLY, never cost/spend figures.
    if dataset_url:
        print(f'Dataset: {dataset_url}')
    # The run's own pages on the platform, one line each, in the spelling the plugin skill parses:
    # ``Experiment: <url>`` and ``Trace: <scenario_folder> <url>``. A link that did not resolve
    # prints no line. Links only, like the line above.
    if run_links is not None:
        if run_links.experiment_url:
            print(f'Experiment: {run_links.experiment_url}')
        for folder, trace_url in sorted(run_links.traces):
            print(f'Trace: {folder} {trace_url}')


def _user_sim_kinds(block: Any, channel: str = 'anomalies') -> set[str]:
    """The KINDS a row's block names on one per-turn channel (never the per-turn detail).

    The block is ``{clean, anomaly_turn_count, anomalies: {"<turn>": {<kind>: <detail>}}}`` —
    the backend's own shape, read and never re-derived — and ``self_report`` is the same map for the
    simulator's own ledger notes, so ``channel`` picks which one. What a summary line can usefully
    carry is which kinds occurred; the turns and the details stay in the file. Shape-tolerant on
    purpose: the vocabulary is the backend's and will grow, so anything unreadable contributes
    nothing rather than raising inside a report about a finished run.
    """
    if not isinstance(block, dict):
        return set()
    per_turn_map = block.get(channel)
    if not isinstance(per_turn_map, dict):
        return set()
    return {str(kind) for per_turn in per_turn_map.values() if isinstance(per_turn, dict) for kind in per_turn}


def _print_verdict(verdict: _RunVerdict, *, dataset_url: str | None, run_links: RunLinks | None = None, stamp_host_toolchain: bool = True) -> None:
    """Print a resolved verdict — the ONE projection of :class:`_RunVerdict` onto the screen.

    Both strategies (native gather, vendor experiment) end here, so a field added to the verdict
    reaches both terminals or neither.

    ``stamp_host_toolchain``: a trial of a ``--trials N`` run leaves the host-toolchain stamp to the
    run's own headline (:func:`_print_trials`), so a run over a host with gaps carries it once.
    """
    _print_summary(
        total=verdict.total,
        failures=verdict.failures,
        incomplete=verdict.incomplete,
        succeeded=verdict.succeeded,
        dataset_url=dataset_url,
        run_links=run_links,
        scored=verdict.scored,
        exit_code_covers=verdict.exit_code_covers,
        ran_count=len(verdict.ran),
        graded_count=len(verdict.graded),
        eval_set_attached=verdict.eval_set_attached,
        # The MAP, not a count: the reasons need different remedies.
        ended_early=verdict.ended_early,
        # And the cause beneath the reason, for the same reason one level down: a give-up is
        # routed by what the simulator SAW, and the reason alone does not say.
        ended_early_causes=verdict.ended_early_causes,
        ended_early_causes_unrecorded=verdict.ended_early_causes_unrecorded,
        degraded_count=len(verdict.degraded),
        # Rows vs TURNS: one emptied turn in twelve and twelve in twelve are different runs,
        # and the line names both numbers.
        degraded_turns=sum(verdict.degraded.values()),
        # The MAP: the line names the tools, and a row count alone cannot.
        world_unread=verdict.world_unread,
        # The published record, so the screen says exactly what the file does.
        host_toolchain_stamp=_LAST_HOST_TOOLCHAIN_RECORD.get('stamp') if stamp_host_toolchain else None,
        # Read from the run's ledger (``_run_ledgers()``), exactly as ``run_summary.json``'s own
        # ``untrustworthy_scenarios`` key is (see ``_write_run_summary``): one source, so the
        # screen and the file cannot disagree about which rows carry the caveat.
        untrustworthy=dict(_run_ledgers().untrustworthy),
        # The MAP again, not a count: the screen speaks only for the rule that is a defect, and
        # only this shape can tell the two apart.
        echoes_dropped=verdict.echoes_dropped,
        # Only the rows that reported a BREACH. A clean block is not news.
        unclean_user_sim={sid: block for sid, block in verdict.user_sim_integrity.items() if _user_sim_breached(block)},
        # The rows the check could not READ — a coverage gap with its own line, never a breach.
        unchecked_user_sim={sid: block for sid, block in verdict.user_sim_integrity.items() if _user_sim_unchecked(block)},
        # Published tables only (the key present): a non-empty mapping is named by kind; a non-mapping is named as unreadable.
        noted_user_sim={
            sid: block
            for sid, block in verdict.user_sim_integrity.items()
            if 'self_report' in block and (not isinstance(block['self_report'], dict) or block['self_report'])
        },
        turn_cap_rerun_count=len(verdict.turn_cap_reruns),
        turn_cap_rerun_restored=_restored_rerun_count(verdict.turn_cap_reruns),
        scenario_verdicts=verdict.scenario_verdicts,
        scenario_high_severity=verdict.scenario_high_severity,
        ended_early_routes=verdict.ended_early_routes,
        measurement_clean=verdict.measurement_clean,
        measurement_unclean_because=verdict.measurement_unclean_because,
    )


def _resolve_dataset_url(sink: ObservabilitySink, dataset_name: str) -> str | None:
    """Best-effort: ask the active sink for its platform dataset URL.

    ``isinstance``-gated on the OPTIONAL :class:`DatasetUrlSink` role (a sink that
    doesn't model a dataset URL — or Braintrust, whose vendor SDK already prints its
    own experiment permalink — is simply skipped) and try-wrapped so a misbehaving
    sink can NEVER fail a finished run: any error degrades to ``None`` ⇒ no link line.
    """
    if not isinstance(sink, DatasetUrlSink):
        return None
    try:
        return sink.dataset_url(dataset_name)
    except Exception:  # noqa: BLE001 — the link is decoration; a sink failure must never fail the run
        return None


def _resolve_run_links(sink: ObservabilitySink, dataset_name: str) -> RunLinks | None:
    """Best-effort: ask the active sink for the links to its experiment and per-scenario traces.

    ``isinstance``-gated on the OPTIONAL :class:`RunLinksSink` role and try-wrapped like
    :func:`_resolve_dataset_url`: a sink without the role, or one that raises, yields ``None`` and
    the summary prints no link lines. Resolved before ``sink.close()`` — a closed client may stop
    serving reads.
    """
    if not isinstance(sink, RunLinksSink):
        return None
    try:
        links = sink.run_links(dataset_name)
    except Exception:  # noqa: BLE001 — the links are decoration; a sink failure must never fail the run
        return None
    return links if links is not None and (links.experiment_url or links.traces) else None


def _merge_run_links(first: RunLinks | None, second: RunLinks | None) -> RunLinks | None:
    """The links of two passes over one trial: the first experiment URL found, and each scenario's latest trace."""
    if first is None or second is None:
        return first or second
    return RunLinks(
        experiment_url=first.experiment_url or second.experiment_url,
        traces=tuple(sorted({**dict(first.traces), **dict(second.traces)}.items())),
    )


async def _read_run_scope(cfg: _RunConfig, settings: Settings) -> tuple[str, list[str]] | int:
    """The run's ``(dataset name, scenario ids)`` once the backend has answered the scope read and the eval-set check, or the
    exit code of the stop the first read ended in: a credit refusal, an outage (the unavailable stop), or a refusal of the
    set. Both strategies read the same two things first, so an outage or refusal is one stop wherever it lands."""
    # Refuse a known-bad eval set BEFORE the dataset pre-flight and the gather, so not one agent subprocess is spent
    # learning what one cheap GET already knows.
    try:
        dataset_name, scenario_ids = await _resolve_run_scope(cfg, settings)
        eval_set_message = await _preflight_eval_set(cfg, settings)
    except CreditRefused as exc:
        return _publish_credit_stop_before_any_scenario(cfg, exc, settings)
    except httpx.HTTPError as exc:
        return _stop_for_outage_before_any_scenario(cfg, exc)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        _publish_early_exit(cfg, exit_codes.REFUSED)
        return exit_codes.REFUSED
    if eval_set_message is not None:
        print(f'ERROR: {eval_set_message}', file=sys.stderr)
        _publish_early_exit(cfg, exit_codes.REFUSED)
        return exit_codes.REFUSED
    return dataset_name, scenario_ids


async def _resolve_run_scope(cfg: _RunConfig, settings: Settings) -> tuple[str, list[str]]:
    """Resolve ``(dataset_name, scenario_ids)`` from ONE key-free ``GET /scenario-sets/{id}``.

    dataset_name — the backend set's human dataset name: the SAME name the
    dataset-upload path targets, so per-scenario tracing (``prepare_run``) and the
    set-level experiment lifecycle register against the ONE provider dataset the
    upload creates — that shared identity is what makes the run appear on the
    uploaded dataset's Experiments tab (LangSmith) / Dataset Runs (Langfuse).
    Best-effort: a fetch failure degrades to the scenario-set id so identity
    resolution can never block a run. The one exception is a 404,
    which raises ``RuntimeError`` whatever the ids: the backend answered and does not hold
    the set, so this runner points at a different backend than the one that generated it,
    and every thread would die on it after the run had started spending.

    scenario_ids — the explicit ``--scenario-id`` list when given; otherwise EVERY
    scenario in the set (the server's folder names, the same ids ``POST /threads``
    resolves). Run-all with an unreachable or empty set raises ``RuntimeError`` —
    silently running zero scenarios would report a clean 0/0 success.
    """
    fetched = None
    try:
        client = build_default_client(settings)
        try:
            fetched = await client.fetch_scenario_set(cfg.scenario_set_id)
        finally:
            await client.aclose()
    except CreditRefused:
        # The organization cannot run work at all; the caller stops the run for it.
        raise
    except httpx.HTTPError as exc:
        _reraise_outage(exc)  # the read waited out the transient budget; the caller stops the run as unavailable
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            raise RuntimeError(
                f'{settings.endpoint.url} does not hold scenario set {cfg.scenario_set_id} (HTTP 404): this runner points at a '
                'different backend than the one that generated it. Log in to the generating backend (`plurai login` with PLURAI_URL naming it, '
                f'or export {URL_ENV} and {TOKEN_ENV}) and retry.'
            ) from exc
        fetched = None
    except Exception:  # noqa: BLE001 — best-effort for identity; fatal below only for run-all
        fetched = None
    dataset_name = (fetched.dataset_name if fetched else None) or cfg.scenario_set_id
    if cfg.scenario_ids:
        _announce_replay_plan_gaps(cfg, cfg.scenario_ids)
        return dataset_name, cfg.scenario_ids
    if fetched is None:
        raise RuntimeError(
            f'--scenario-id omitted (run the whole set) but scenario set {cfg.scenario_set_id} '
            'could not be fetched to enumerate its scenarios — check the backend configuration (`plurai login --status`), or pass --scenario-id explicitly.'
        )
    scenario_ids = [s.folder_name for s in (fetched.scenarios or []) if s.folder_name]
    if not scenario_ids:
        raise RuntimeError(f'--scenario-id omitted (run the whole set) but scenario set {cfg.scenario_set_id} lists no scenarios — nothing to run.')
    _announce_replay_plan_gaps(cfg, scenario_ids)
    return dataset_name, scenario_ids


def _announce_replay_plan_gaps(cfg: _RunConfig, scenario_ids: Sequence[str]) -> None:
    """Print :func:`_replay_plan_gaps`' line once, when the run's scope is known; never a refusal."""
    if (gap := _replay_plan_gaps(cfg, scenario_ids)) is not None:
        print(f'[regressions] WARNING: {gap}', file=sys.stderr)


async def _run_gathered_async(cfg: _RunConfig, settings: Settings, sink: ObservabilitySink) -> int:
    # A previous run's verdict must not survive an early guard exit below.
    _clear_stale_run_summary(cfg.out, partial=cfg.is_partial)
    # ...and neither must a previous IN-PROCESS run's accumulating ledgers (see
    # :func:`_reset_run_ledgers`), which have no per-run recorder of their own to clear them.
    _reset_run_ledgers()
    _prepare_trial_dirs(cfg)
    # The ONE active sink is resolved by ``main()`` and handed in; this strategy OWNS
    # its lifecycle: warm the process-global init ONCE before the gather, close ONCE
    # in the ``finally`` (orchestrators are per-scenario and never close it).
    await sink.warm_once()
    # The exclusive resources' pools, set BEFORE the gather — a single run's or the trials' — so every attempt of
    # every trial leases from the same ones (:data:`_RESOURCE_LEASES`).
    leases_token = _RESOURCE_LEASES.set(_ResourceLeases(cfg.resource_leases) if cfg.resource_leases else None)
    try:
        # The run's dataset identity = the set's human dataset name (same as the upload
        # path), so the experiment lifecycle AND per-scenario tracing register against
        # the one provider dataset the upload creates. The same fetch enumerates the
        # set when --scenario-id was omitted (run the whole set).
        read = await _read_run_scope(cfg, settings)
        if isinstance(read, int):
            return read
        dataset_name, scenario_ids = read
        # Isolation BEFORE anything runs — the only guard here whose failure mode
        # lands outside the test: a simulated user driving a real system. The guard is armed from the
        # declaration; the measurement's findings go through the one gate funnel, right here.
        _preflight_egress(cfg, settings, keep_guard_log=cfg.is_partial)
        if (held_code := _held_by_gates(cfg)) is not None:
            return held_code
        # The suite's PREMISES, after its isolation: a run scored against documents that changed
        # since the scenarios were authored measures the staleness, not the agent. A measured drift
        # on a full run is a holding finding; the one exit funnel decides.
        _preflight_scan_contract(cfg)
        if (held_code := _held_by_gates(cfg)) is not None:
            return held_code
        cfg = _resolve_max_turns(cfg, await _preflight_solvability(cfg, settings))
        run_name = _unique_run_name(cfg, base=dataset_name)
        # PRE-FLIGHT dataset upload — BEFORE the gather, not after it. The dataset defines
        # the experiment's INPUTS, so every example must exist before any scenario opens an
        # experiment session: LangSmith pins a session's example set to the dataset version
        # as of session creation, and neither reference_dataset_id nor its version can be set
        # post-hoc — a session opened against an empty dataset shows "No results found"
        # forever. The arize/experiment strategy uploads pre-run
        # for this same reason (see _setup_experiment); the gather path does the same.
        #
        # Two independent triggers, deliberately scored differently:
        # * ``wants`` — the operator passed --upload-datasets. Contributes to the exit code.
        # * ``needs`` — the SINK declares it cannot render a correct experiment unless its
        #   dataset is populated first (``requires_preflight_dataset``; LangSmith sets it).
        #   An implicit, sink-required pre-flight must NEVER change the exit code: the
        #   operator did not ask for an upload, so its failure is not their failure — the run
        #   degrades to per-scenario example creation.
        # Read via getattr so this stays a one-sink concern: no Protocol change, no isinstance,
        # no matching on ``sink.name``, and every other sink resolves to False untouched.
        #
        # No outcome gate applies: there are no outcomes before the gather, and the dataset's
        # content is the INPUT scenarios fetched from the server — independent
        # of run outcomes, and a 0/N run is precisely when the grid most needs to render so
        # the failures are visible. Repeat uploads are idempotent via dataset_upload_metadata.json.
        upload_ok = True
        preflight_degraded = False
        wants_upload = cfg.upload_datasets
        needs_preflight = bool(getattr(sink, 'requires_preflight_dataset', False))
        if wants_upload or needs_preflight:
            upload = await _upload_datasets(cfg, settings, sink, scenario_ids=scenario_ids)
            upload_ok = bool(upload and upload.ok) if wants_upload else True
            # For a sink that REQUIRES the pre-flight, "uploaded nothing" is as broken as
            # an outright failure even though it reports ok=True (an empty or wholly
            # incomplete set) — the dataset the experiment pins to is still empty.
            preflight_degraded = needs_preflight and not (upload and upload.ok and upload.uploaded)
        if cfg.trials > 1:
            # N complete runs of the set, CONCURRENTLY — wall time is one sample's, not N (see _run_trials).
            trial_codes = await _run_trials(cfg, settings, scenario_ids=scenario_ids, dataset_name=dataset_name, run_name=run_name, upload_ok=upload_ok)
        else:
            failures, rows = await _gather_run(cfg, settings, sink, scenario_ids=scenario_ids, dataset_name=dataset_name, run_name=run_name)
        # Resolved BEFORE the close (a drained/shut-down sink client may stop
        # serving API reads); best-effort — None just drops the summary's link line.
        dataset_url = _resolve_dataset_url(sink, dataset_name)
        run_links = _resolve_run_links(sink, dataset_name)
    finally:
        _RESOURCE_LEASES.reset(leases_token)
        # This strategy holds the sink for its whole run — it owns the ONE close.
        await sink.close()

    if isinstance(abort := _run_abort(), _MockAbort):
        exit_code = _publish_mock_abort(cfg, abort, run_name=run_name)
    elif isinstance(abort, _MeasuredStop):
        # Whatever else this invocation found (a failed upload, a worse trial code), a credit, capacity or unavailable stop measured nothing more:
        # the exit is "not measured", and the trials' ``max`` never gets to outrank it. A trials run keeps each trial's own
        # summary and publishes no cross-trial aggregate: a partial set of trials is a different claim.
        exit_code = exit_codes.NOT_MEASURED
        if cfg.trials == 1:
            _print_verdict(
                _write_gathered_summary(cfg, exit_code=exit_code, run_name=run_name, rows=rows, aborted=abort.to_record()),
                dataset_url=dataset_url,
                run_links=run_links,
            )
        else:
            _stamp_credit_abort(cfg, abort, run_name=run_name)
        _print_measured_stop(abort)
    elif cfg.trials > 1:
        exit_code = _publish_trials(cfg, run_name=run_name, scenario_ids=scenario_ids, trial_codes=trial_codes, dataset_url=dataset_url)
    else:
        # Computed BEFORE the summary so the file records exactly the code this call returns.
        exit_code = _gathered_exit_code(failures=bool(failures), upload_ok=upload_ok)
        verdict = _write_gathered_summary(cfg, exit_code=exit_code, run_name=run_name, rows=rows)
        # Printed FROM the written verdict, so the screen and the file can never disagree.
        _print_verdict(verdict, dataset_url=dataset_url, run_links=run_links)
    if preflight_degraded:
        # Printed HERE, after the summary, because the pre-flight line itself scrolls
        # off behind N scenarios of output — and because the upload's own status line says
        # only that an upload failed, never what it COSTS. For a sink that requires the
        # pre-flight the consequence is a partly- or wholly-empty Experiments grid for a
        # run that otherwise reports success, which is precisely the silent failure this
        # note exists to surface.
        print(
            '[dataset-upload] WARNING: this sink needs its dataset populated BEFORE the run, and the '
            'pre-flight did not complete. Scenarios whose example was created after the experiment '
            'session opened will be MISSING from the Experiments grid — the run above is scored '
            'correctly, but the grid under-reports it. Re-run once the upload succeeds (pass '
            '--upload-datasets to make this failure fail the run).',
            file=sys.stderr,
        )
    return exit_code


async def _gather_run(
    cfg: _RunConfig, settings: Settings, sink: ObservabilitySink, *, scenario_ids: list[str], dataset_name: str, run_name: str
) -> tuple[int, dict[str, Any]]:
    """ONE run's bounded gather over ``scenario_ids`` into ``cfg.out``: ``(failures, the rows)``.

    The rows are :func:`_write_run_summary`'s per-row keyword arguments, written by
    :func:`_write_gathered_summary`. Shared by a single run and by every trial body of a
    ``--trials N`` run (:func:`_run_trials`), which is what keeps a trial a run of a single run's shape.
    """
    # The ceiling-hit re-run ledger (--rerun-on-turn-cap): filled per scenario by _run_one,
    # which is the only frame that sees both attempts. ``timings`` rides the same way: only
    # that frame brackets a whole row, and a failed row must still report its clock.
    turn_cap_reruns: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}
    # The stability ledger + its run-level pool, sized HERE because this is
    # the frame that knows how many scenarios there are.
    stability: dict[str, dict[str, Any]] = {}
    stability_budget = _stability_budget_for(scenario_ids, cfg.run_phase, sample_kind=cfg.stability_sample_kind)
    async with experiment_scope(sink, dataset_name=dataset_name, run_name=run_name):
        sem = asyncio.Semaphore(max(1, cfg.concurrency))

        async def _bounded(scenario_id: str) -> ThreadResult | None:
            # Acquired by hand so the row can free its slot early (a judge replay, see
            # :func:`_drive_scenario_with_stability`); idempotent, so the ``finally`` is always safe.
            await sem.acquire()
            held = True

            def release_slot() -> None:
                nonlocal held
                if held:
                    held = False
                    sem.release()

            try:
                return await _run_one(
                    cfg,
                    settings,
                    sink,
                    scenario_id=scenario_id,
                    dataset_name=dataset_name,
                    run_name=run_name,
                    turn_cap_reruns=turn_cap_reruns,
                    timings=timings,
                    stability_budget=stability_budget,
                    stability=stability,
                    release_slot=release_slot,
                )
            finally:
                release_slot()

        # _run_one isolates per-scenario failure (writes error.json, returns None), so the
        # ordinary crash never poisons its siblings. ``return_exceptions`` is for what that
        # handling cannot reach: anything raised OUTSIDE it — in the bracket itself, in the
        # semaphore, in the ledger writes. Without it, ONE such raise propagates out of the
        # gather, abandons every sibling mid-flight and skips _write_run_summary entirely, so
        # a run that scored six of seven scenarios leaves no run_summary.json at all. The
        # rows are normalized back to None (a failed row) before anything reads them, because
        # an exception object is TRUTHY and the counts below are built from truthiness.
        # LAUNCHED in the trial's own order (:func:`_launch_order`) — the semaphore admits tasks in the
        # order they first run — but gathered back in list order, so every row reader below is unchanged.
        launched = {index: asyncio.ensure_future(_bounded(scenario_ids[index])) for index in _launch_order(len(scenario_ids), cfg.trial)}
        settled = await asyncio.gather(*(launched[index] for index in range(len(scenario_ids))), return_exceptions=True)
        results = [_row_result(sid, outcome) for sid, outcome in zip(scenario_ids, settled, strict=True)]
        # Paired ONCE, because every per-row reader below wants the same (id, result) rows
        # and a zip is a one-shot iterator. A ``list`` rather than seven fresh zips: they
        # must all see the same pairing, and ``strict=True`` spelled seven times is seven
        # chances for one copy to lose it and silently truncate to the shorter side.
        rows = list(zip(scenario_ids, results, strict=True))
        failures = sum(1 for ok in results if not ok)
        _note_host_effects(rows)
        _note_tool_channel(rows, cfg.agent_module_path)
        _note_mock_unanswered(rows)
        _note_ipc_trimmed(rows)
        return failures, {
            # A ThreadResult is truthy, None falsy — the same truthiness the failure
            # count above is built from, so the map and the count cannot disagree.
            'scenarios': {sid: _row_outcome(ok, sid) for sid, ok in rows},
            # The grading EVIDENCE, read off the same results: which scenarios actually
            # produced a policy verdict. Collected here because this is the only place
            # holding the finalize payloads (the summary sees strings, not results).
            'graded': _graded_ids(cfg, rows),
            # Same reason, for the rows the harness cut short. The outcome map gives a cut
            # scenario its own 'incomplete' status, but the REASON lives only here — and a row that
            # ended early for any OTHER reason is still an 'ok' whose evals are low for turns it
            # never took.
            'ended_early': _ended_early_ids(rows),
            'ended_early_causes': _ended_early_causes(rows),
            'plan_exhausted_disputed_turns': _plan_exhausted_disputed_turns(rows),
            'agent_close_preempted_targets': _agent_close_preempted_targets(rows),
            # Same reason again, for the apparatus rather than the conversation: the outcome
            # map cannot say that a row's target check died, and nothing downstream can
            # re-derive it from the strings the summary sees.
            'measurement_integrity': _measurement_integrity(rows),
            # The SCENARIO-fidelity half of the same payload — the user simulator's own
            # anomalies. Read here for the same reason: the summary sees strings.
            'user_sim_integrity': _user_sim_integrity(rows),
            'eval_revisions': _eval_revisions(rows),
            # The per-family gradient, off the same payloads and for the same reason.
            'family_gradient': _family_gradient(_family_row(result) for _sid, result in rows),
            # And the client-side integrity rows: which scenarios carry turns the HARNESS
            # emptied. Only this side knows — the backend graded a well-formed conversation.
            'degraded': _degraded_ids(rows),
            # And the other breach only this side can see: tools that asked for a seeded record and
            # came back empty. The seeded world and every tool response meet only in these rows.
            'world_unread': _world_unread_ids(rows, reader_scopes=_reader_scopes(Path(cfg.agent_module_path).parent)),
            # Its sibling, for the OTHER thing only this side knows about that channel: which
            # scenarios had messages taken out of it by the single-role split, and by which rule.
            'echoes_dropped': _echoes_dropped_ids(rows),
            'scenario_verdicts': _scenario_verdict_ids(rows),
            'scenario_high_severity': _scenario_high_severity_ids(rows),
            'turn_cap_reruns': turn_cap_reruns,
            'timings': timings,
            'stability': stability,
            'not_run': sorted(sid for sid, result in rows if _not_run(sid, result)),
        }


def _write_gathered_summary(
    cfg: _RunConfig, *, exit_code: int, run_name: str, rows: Mapping[str, Any], egress_dir: Path | None = None, aborted: Mapping[str, Any] | None = None
) -> _RunVerdict:
    """:func:`_write_run_summary` for one gathered run (:func:`_gather_run`'s rows) into ``cfg.out``."""
    return _write_run_summary(
        cfg.out,
        exit_code=exit_code,
        scenarios=rows['scenarios'],
        graded=rows['graded'],
        cfg=cfg,
        run_name=run_name,
        ended_early=rows['ended_early'],
        ended_early_causes=rows['ended_early_causes'],
        measurement_integrity=rows['measurement_integrity'],
        user_sim_integrity=rows['user_sim_integrity'],
        eval_revisions=rows['eval_revisions'],
        degraded=rows['degraded'],
        echoes_dropped=rows['echoes_dropped'],
        turn_cap_reruns=rows['turn_cap_reruns'],
        timings=rows['timings'],
        stability=rows['stability'],
        family_gradient=rows['family_gradient'],
        world_unread=rows['world_unread'],
        scenario_verdicts=rows['scenario_verdicts'],
        scenario_high_severity=rows['scenario_high_severity'],
        plan_exhausted_disputed_turns=rows['plan_exhausted_disputed_turns'],
        agent_close_preempted_targets=rows['agent_close_preempted_targets'],
        egress_dir=egress_dir,
        aborted=aborted,
        not_run=rows['not_run'],
    )


def _trial_out(cfg: _RunConfig, index: int) -> Path:
    """Trial ``index``'s run directory: ``$OUT`` for trial 1, ``$OUT/trial_<index>`` otherwise."""
    return cfg.out / trial_dir_name(index)


def _prepare_trial_dirs(cfg: _RunConfig) -> None:
    """At strategy entry: clear each trial run's previous verdict the way ``$OUT``'s is cleared,
    and MOVE every ``trial_<k>`` directory this run will not write (``k > --trials``) under
    ``previous_attempt/``. Readers enumerate trials only from ``run_summary.trials.dirs``; a trial
    an older run left behind must not sit in ``$OUT`` reading as one of this run's."""
    for index in range(2, cfg.trials + 1):
        _clear_stale_run_summary(_trial_out(cfg, index), partial=cfg.is_partial)
    try:
        stale = sorted(
            p for p in cfg.out.glob(f'{TRIAL_DIR_PREFIX}*') if (k := p.name.removeprefix(TRIAL_DIR_PREFIX)).isdigit() and int(k) > cfg.trials and p.is_dir()
        )
    except OSError:
        return
    for path in stale:
        target = cfg.out / PREVIOUS_ATTEMPT_DIRNAME / path.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            path.replace(target)
            print(f'[trials] {path.name}/ is not a trial of this run; moved to {PREVIOUS_ATTEMPT_DIRNAME}/{path.name}/', file=sys.stderr)
        except OSError as exc:
            print(f"[trials] WARNING: could not move the stale {path.name}/ aside ({exc}); it is NOT one of this run's trials.", file=sys.stderr)


def _trials_out_refusal(cfg: _RunConfig) -> str | None:
    """REFUSE a ``--scenario-id`` re-run whose ``--trials`` differs from the run it merges into.

    A targeted re-run is a correction to the population already in ``--out`` and merges into each
    trial run's summary; merged at a different N, the corrected rows and the carried ones would be
    samples of different sizes under one ``trials`` block.
    """
    if not cfg.is_partial:
        return None
    prior = _read_json_object(cfg.out / RUN_SUMMARY_FILENAME)
    if prior is None:
        return None
    block = prior.get('trials')
    prior_n = block.get('n') if isinstance(block, dict) and block.get('schema') == TRIALS_SCHEMA else 1
    if prior_n == cfg.trials:
        return None
    return f'{cfg.out} holds a run of {prior_n} trial(s) per scenario, and a --scenario-id re-run merges into it; re-run with --trials {prior_n}.'


async def _settled[T](awaitable: Awaitable[T]) -> T | BaseException:
    """One awaitable's outcome as ``asyncio.gather(..., return_exceptions=True)`` hands back an
    ``Exception``; cancellation and the other ``BaseException``\\ s propagate (the run is ending)."""
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 — returned, as gather returns it; the caller types it
        return exc


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


#: The run-log line an external-world run prints once, before its first attempt.
EXTERNAL_WORLD_SERIAL_LINE = '[world] external world: trials and scenarios run one at a time'
#: The same line for an agent-owned store declared ``world_state.slots: 1``.
AGENT_WORLD_SERIAL_LINE = '[world] agent world: trials and scenarios run one at a time'
_WORLD_SERIAL_LINES = {WORLD_OWNER_EXTERNAL: EXTERNAL_WORLD_SERIAL_LINE, WORLD_OWNER_AGENT: AGENT_WORLD_SERIAL_LINE}


def _external_world_slots_line(slots: int, width: int, concurrency: int, *, owner: str) -> str:
    """The run-log line a SLOTTED run prints once, before its first attempt; ``owner`` is the world's owner word."""
    clamp = f', --concurrency {concurrency}' if width < slots else ''
    return f'[world] {owner} world: up to {width} at once, one world copy each ({slots} slots{clamp})'


def _one_at_a_time_for_world(cfg: _RunConfig, world_owner: tuple[str, Path] | None, world_slots: int | None = None) -> _RunConfig:
    """``cfg`` with every attempt bounded by the world copies when the bench declares ``world_state.owner: external``,
    or ``world_state.owner: agent`` beside a ``world_state.slots`` count.

    Such a world is a store outside every per-scenario child — the child calls ``reset()`` at each
    attempt's init and ``world_snapshot()`` at its finalize — so two attempts in flight on ONE copy
    wipe each other's in-flight writes and attribute each other's writes.

    The EXPERIMENT strategy's bound; the gathered strategy leases the same copies instead (:func:`_lease_resources`).

    * ONE copy (external with ``world_slots`` undeclared or 1, agent with 1): ``concurrency`` drops to 1
      (one scenario at a time per run) and ``one_at_a_time`` makes
      :func:`_run_trials` run the trials one after another.
    * N copies (``world_slots: N >= 2``, :func:`~plurai.orchestrator.orchestrator._declared_world_slots`):
      ``concurrency`` is ``min(C, N)``, the trials still run one after another, and every attempt leases
      one copy from ``cfg.world_slots`` for its child's life — so attempts in flight never outnumber
      copies and no copy has two at once.

    Said ONCE in the run log. A mock-owned world (each scenario its own seeded session) and an
    agent-owned one with NO declared count (a store each child holds in its own process) keep the
    parallelism they were given. Only the dialogs are bounded: a ``judge_replay`` frees its run slot
    first and may overlap the next dialog, because it rescores saved artifacts server-side and never
    reads the world (it holds no world copy either).
    """
    if world_owner is None or world_owner[0] not in _WORLD_SERIAL_LINES:
        return cfg
    owner = world_owner[0]
    if owner == WORLD_OWNER_AGENT and world_slots is None:
        return cfg
    if world_slots is None or world_slots < 2:
        print(_WORLD_SERIAL_LINES[owner])
        return dataclasses_replace(cfg, concurrency=1, one_at_a_time=True)
    width = min(max(1, cfg.concurrency), world_slots)
    print(_external_world_slots_line(world_slots, width, cfg.concurrency, owner=owner))
    return dataclasses_replace(cfg, concurrency=width, one_at_a_time=True, world_slots=_WorldSlots(world_slots))


def _one_at_a_time_for_single_instance(cfg: _RunConfig, resources: Sequence[str]) -> _RunConfig:
    """``cfg`` with ONE agent process at a time when the bench declares ``harness.single_instance`` resources
    (a fixed port the agent binds at setup, a lock it takes): two processes side by side would collide on it, and
    the second's setup fails for a reason that is ours, not the agent's. ``concurrency`` drops to 1 and the trials
    run one after another. Said once in the run log; a bench that declares none keeps its parallelism."""
    if not resources or (cfg.concurrency <= 1 and (cfg.trials < 2 or cfg.one_at_a_time)):
        return cfg
    print(f'[harness] single-instance resource ({", ".join(resources)}): trials and scenarios run one at a time')
    return dataclasses_replace(cfg, concurrency=1, one_at_a_time=True)


def _world_lease_line(owner: str, copies: int) -> str:
    """The run-log line a gathered run that leases a bounded world prints once, before its first attempt."""
    if copies == 1:
        return f'[world] {owner} world: one copy, held by one agent process at a time; the other scenarios and trials wait for it'
    return f'[world] {owner} world: {copies} copies, one agent process on each at a time; the other scenarios and trials wait for a free copy'


def _single_instance_lease_line(resources: Sequence[str]) -> str:
    """The run-log line a gathered run that leases ``harness.single_instance`` resources prints once."""
    return f'[harness] single-instance resource ({", ".join(resources)}): held by one agent process at a time; the other scenarios and trials wait for it'


def _lease_resources(cfg: _RunConfig, world_owner: tuple[str, Path] | None, world_slots: int | None, single_instance: Sequence[str]) -> _RunConfig:
    """``cfg`` carrying one lease pool per exclusive resource, for the GATHERED strategy; ``concurrency`` and the trials
    are untouched (the experiment strategy keeps :func:`_one_at_a_time_for_world` and
    :func:`_one_at_a_time_for_single_instance` instead).

    * A world the runner bounds — ``owner: external``, or ``owner: agent`` beside a ``slots`` count, the same rule as
      :func:`_one_at_a_time_for_world` — is ``slots`` copies (one undeclared); a holder of one of 2+ copies is told
      which (``PLURAI_WORLD_SLOT``).
    * Each ``harness.single_instance`` resource is one copy.

    Every attempt holds one copy of each from before its thread opens until its agent process closed, and gives them
    back BEFORE its ``/finalize`` (``on_child_closed``): no two agent processes share a copy, while finalizes, judge
    replays and the trials overlap the next dialog. Said once in the run log."""
    specs: list[_LeaseSpec] = []
    if world_owner is not None and world_owner[0] in _WORLD_SERIAL_LINES and not (world_owner[0] == WORLD_OWNER_AGENT and world_slots is None):
        copies = world_slots if world_slots is not None and world_slots >= 2 else 1
        specs.append(_LeaseSpec(f'{world_owner[0]} world', copies, names_slot=copies >= 2))
        print(_world_lease_line(world_owner[0], copies))
    if single_instance:
        specs.extend(_LeaseSpec(f'single-instance resource ({resource})', 1) for resource in single_instance)
        print(_single_instance_lease_line(single_instance))
    return dataclasses_replace(cfg, resource_leases=tuple(specs)) if specs else cfg


def _alive_cap(cfg: _RunConfig) -> int | None:
    """At most how many agent processes a gathered run's leases let live at once (the fewest copies), ``None`` with none."""
    return min((spec.copies for spec in cfg.resource_leases), default=None)


def _sibling_width(cfg: _RunConfig) -> int:
    """How many agent processes may have run beside an attempt's, for the re-drive rules that ask whether a sibling
    could have taken its memory: ``--concurrency``, but 1 under a one-copy lease, where no sibling process lives."""
    return 1 if _alive_cap(cfg) == 1 else cfg.concurrency


#: The per-child memory budget, sized for the heaviest agent frameworks (LlamaIndex / Mastra agent processes): what one
#: in-flight attempt is budgeted when the fan-out is sized against the sandbox's memory limit.
_CHILD_MB = 320
#: What the runner, the plugin session and the OS keep of that limit before any child is counted.
_RESERVE_MB = 1024
#: Where the kernel exposes this process's cgroup (v2 at the root, v1 under ``memory/``).
_CGROUP_ROOT = Path('/sys/fs/cgroup')
#: cgroup v1 reports "no limit" as a page-rounded int64 max (9223372036854771712); anything this large is none.
_CGROUP_V1_UNLIMITED = 1 << 60


def _memory_limit_mb(root: Path = _CGROUP_ROOT) -> int | None:
    """The memory limit this process runs under, in MiB, or ``None`` when none is known.

    cgroup v2 ``memory.max`` (``max`` → none), else cgroup v1 ``memory/memory.limit_in_bytes`` (the
    unlimited sentinel → none). Neither file exists on macOS or on a v2 host outside a container, so
    that is ``None``. An unreadable or non-integer file is ``None``, never an error.
    """
    for path in (root / 'memory.max', root / 'memory' / 'memory.limit_in_bytes'):
        try:
            text = path.read_text(encoding='ascii').strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text == 'max':
            return None
        try:
            limit = int(text)
        except ValueError:
            return None
        return None if limit <= 0 or limit >= _CGROUP_V1_UNLIMITED else limit // (1 << 20)
    return None


def _trials_in_turn_for_memory(cfg: _RunConfig, limit_mb: int | None) -> _RunConfig:
    """``cfg`` with its trials run one after another when ``trials × concurrency`` agent processes do
    not fit the known memory limit (``T·C·_CHILD_MB > limit − _RESERVE_MB``); otherwise ``cfg`` itself.

    ``concurrency`` is never changed — each trial still runs C scenarios at once — so a world-copy
    clamp set by :func:`_one_at_a_time_for_world` holds; a run already in turn is left alone (and
    prints no second line). Verdicts are unchanged; the wall time is about ×T. Said once in the run log.

    What is counted is the agent processes that can be ALIVE: under leases (:func:`_lease_resources`) the
    scenarios waiting for a copy have no process yet, so at most the fewest copies are.
    """
    width = max(1, cfg.concurrency)
    alive = cfg.trials * width
    if (cap := _alive_cap(cfg)) is not None:
        alive = min(alive, cap)
    if cfg.trials < 2 or cfg.one_at_a_time or limit_mb is None or alive * _CHILD_MB <= limit_mb - _RESERVE_MB:
        return cfg
    print(f'[trials] memory limit {limit_mb} MB: {cfg.trials} × {width} agent processes do not fit — trials run one after another, {width} at a time')
    return dataclasses_replace(cfg, one_at_a_time=True)


async def _run_trials(cfg: _RunConfig, settings: Settings, *, scenario_ids: list[str], dataset_name: str, run_name: str, upload_ok: bool) -> list[int | None]:
    """Run the set ``cfg.trials`` times, CONCURRENTLY, as that many complete runs; their exit codes,
    ``None`` for a trial that raised outside its scenarios (it wrote no summary of this run).

    Each trial is a run of a single run's shape (:func:`_run_trial`), so every per-row ledger, re-drive,
    turn-cap re-run and ``--scenario-id`` merge stays correct per trial. The trials run at the
    same time — ``--concurrency`` keeps its meaning (scenarios in flight PER RUN), so the load is
    C x N agent processes and the wall time is about one sample's. Their SETUPS share one gate of C
    (:data:`_SETUP_GATE`), and each trial launches its list from its own offset (:func:`_launch_order`).

    An external world (``cfg.one_at_a_time``, :func:`_one_at_a_time_for_world`) runs them one after
    another instead, each at ``cfg.concurrency``: 1 against one shared world, ``min(C, N)`` against N
    world copies (``world_state.slots``). So does a run whose C x N agent processes do not fit the
    sandbox's memory limit (:func:`_trials_in_turn_for_memory`), each trial at ``cfg.concurrency``.
    A gathered run's exclusive resources are leases instead (:func:`_lease_resources`): the trials still run
    side by side, and only the agent processes wait for a free copy.
    """
    width = max(1, cfg.concurrency)
    if cfg.one_at_a_time:
        copies = ', one world copy each' if cfg.world_slots is not None else ''
        at_once = 'one agent process at a time' if width == 1 else f'{width} agent processes at a time{copies}'
        print(f'[trials] trials {cfg.trials}, one after another ({at_once})')
    else:
        alive = '' if (cap := _alive_cap(cfg)) is None else f', at most {min(cap, cfg.trials * width)} alive at once by lease'
        print(f'[trials] trials {cfg.trials} × concurrency {width} = {cfg.trials * width} agent processes (at most {width} in setup at once{alive})')
    if cfg.stability_retry_enabled:
        print(
            '[trials] the stability retry is OFF for this run: the other trials are the second sample '
            "(a crash takes the re-drive's one attempt; a run the user simulator voided is still re-run)."
        )
    # Set BEFORE the gather, so every trial task — and every scenario task it spawns — inherits the one gate.
    gate_token = _SETUP_GATE.set(asyncio.Semaphore(width))
    try:
        trials = (
            _run_trial(cfg, settings, index=index, scenario_ids=scenario_ids, dataset_name=dataset_name, run_name=run_name, upload_ok=upload_ok)
            for index in range(1, cfg.trials + 1)
        )
        settled = [await _settled(trial) for trial in trials] if cfg.one_at_a_time else await asyncio.gather(*trials, return_exceptions=True)
    finally:
        _SETUP_GATE.reset(gate_token)
    if not cfg.one_at_a_time and _run_abort() is None:
        settled = await _rerun_setup_collisions(
            cfg, settings, list(settled), scenario_ids=scenario_ids, dataset_name=dataset_name, run_name=run_name, upload_ok=upload_ok
        )
    codes: list[int | None] = []
    for index, outcome in enumerate(settled, start=1):
        if isinstance(outcome, BaseException):
            # Cancellation / KeyboardInterrupt is the RUN ending, not a trial failing (``_row_result``'s rule).
            if not isinstance(outcome, Exception):
                raise outcome
            logger.error('trial %d raised outside its scenarios', index, exc_info=outcome)
            print(
                f'[trials] ERROR: trial {index} raised outside its scenarios ({type(outcome).__name__}: {outcome}); its rows count as not run.', file=sys.stderr
            )
            codes.append(None)
        else:
            exit_code, verdict, trial_links = outcome
            codes.append(exit_code)
            if isinstance(_run_abort(), _MockAbort):
                continue
            print(f'\n[trial {index}/{cfg.trials}] {trial_run_name(run_name, index)} → {_trial_out(cfg, index)}')
            _print_verdict(verdict, dataset_url=None, run_links=trial_links, stamp_host_toolchain=False)
    return codes


def _setup_failures(run_dir: Path, scenario_ids: Sequence[str]) -> list[str]:
    """The scenarios whose attempt in ``run_dir`` failed during SETUP (``error.json`` ``raised_at: init``)."""
    return [sid for sid in scenario_ids if (record := _read_json_object(run_dir / sid / 'error.json')) is not None and record.get('raised_at') == 'init']


async def _rerun_setup_collisions(
    cfg: _RunConfig,
    settings: Settings,
    settled: list[tuple[int, _RunVerdict, RunLinks | None] | BaseException],
    *,
    scenario_ids: list[str],
    dataset_name: str,
    run_name: str,
    upload_ok: bool,
) -> list[tuple[int, _RunVerdict, RunLinks | None] | BaseException]:
    """Trials run side by side can collide on what only one agent process may hold — a fixed port bound at setup
    (EADDRINUSE), a lock — and the loser's setup fails, often untyped (the agent's own setup swallows the errno).
    So each trial's scenarios that failed during SETUP are re-run ONCE, one at a time, after every trial has ended
    (nothing beside them), and merged into that trial's run as a ``--scenario-id`` re-run merges. The trial's exit
    code is the worse of the two. A run whose trials ran one after another never gets here; a run that leases a
    declared resource (:func:`_lease_resources`) does, as the backstop for one the bench did not declare."""
    for index, outcome in enumerate(settled, start=1):
        if isinstance(outcome, BaseException) or not (failed := _setup_failures(_trial_out(cfg, index), scenario_ids)):
            continue
        print(f'[trials] trial {index}: {len(failed)} scenario(s) failed during setup while trials ran side by side — re-running them one at a time')
        serial = dataclasses_replace(cfg, scenario_ids=failed, concurrency=1, one_at_a_time=True)
        trial = _run_trial(serial, settings, index=index, scenario_ids=failed, dataset_name=dataset_name, run_name=run_name, upload_ok=upload_ok)
        rerun = await _settled(trial)
        settled[index - 1] = rerun if isinstance(rerun, BaseException) else (max(outcome[0], rerun[0]), rerun[1], _merge_run_links(outcome[2], rerun[2]))
    return settled


async def _run_trial(
    cfg: _RunConfig, settings: Settings, *, index: int, scenario_ids: list[str], dataset_name: str, run_name: str, upload_ok: bool
) -> tuple[int, _RunVerdict, RunLinks | None]:
    """Trial ``index``: one complete run into :func:`_trial_out`, under its own sink, run name
    (``<run>-t<index>``) and per-row ledgers; returns its exit code, verdict and run links, which
    :func:`_run_trials` prints in trial order. The links are resolved from the trial's own sink
    before it closes, since each trial is its own experiment on the platform.

    Its stability retry is OFF — the other trials are the second sample — while the INVALID-run
    re-run still fires: the budget handed down is real, and only ``--no-stability-retry``'s gate
    (which an invalid run is exempt from) is closed.
    """
    trial_cfg = dataclasses_replace(cfg, out=_trial_out(cfg, index), stability_retry=False, trial=_TrialIdentity(index=index, count=cfg.trials, group=run_name))
    name = trial_run_name(run_name, index)
    token = _TRIAL_LEDGERS.set(_RunLedgers.fresh())
    try:
        sink = create_observability_sink(settings)
        await sink.warm_once()
        try:
            failures, rows = await _gather_run(trial_cfg, settings, sink, scenario_ids=scenario_ids, dataset_name=dataset_name, run_name=name)
            links = _resolve_run_links(sink, dataset_name)
        finally:
            await sink.close()
        if isinstance(abort := _run_abort(), _MockAbort):
            return exit_codes.MOCK_UNANSWERED, _write_mock_abort_summary(trial_cfg, abort, run_name=name), links
        if isinstance(abort, _MeasuredStop):
            # The trial finished or was stopped, and either way the run is "not measured": its code is not a trial's own.
            exit_code = exit_codes.NOT_MEASURED
            return (
                exit_code,
                _write_gathered_summary(trial_cfg, exit_code=exit_code, run_name=name, rows=rows, egress_dir=cfg.out, aborted=abort.to_record()),
                links,
            )
        exit_code = _gathered_exit_code(failures=bool(failures), upload_ok=upload_ok)
        return exit_code, _write_gathered_summary(trial_cfg, exit_code=exit_code, run_name=name, rows=rows, egress_dir=cfg.out), links
    finally:
        _TRIAL_LEDGERS.reset(token)


def _attempt_from_disk(scenario_dir: Path) -> ThreadResult | None:
    """One trial's attempt at a scenario, rebuilt from its ``verdicts.json`` (the verdicts) and ``transcript.json``
    (``finalize_record``, ``ended_early``) for the stability projection — which reads only the finalize payload, the
    thread id and ``ended_early``. ``None`` for an attempt that graded nothing. No derived numbers: the projection reads
    cells, never ``pass_rate``."""
    judged = verdicts.read_or_none(scenario_dir)
    if judged is None or not judged['graded']:
        return None
    doc = _read_json_object(scenario_dir / 'transcript.json') or {}
    metadata: dict[str, Any] = raw if isinstance(raw := doc.get('metadata'), dict) else {}
    record = raw if isinstance(raw := doc.get('finalize_record'), dict) else {}
    try:
        per = verdicts.as_policy_results(judged, record=record)
        finalize = FinalizeResponse(status='completed', aggregate_score=0.0, pass_rate=None, policy_evaluation_results=per)
    except ValidationError:
        return None
    return ThreadResult(
        thread_id=str(metadata.get('thread_id') or judged['attempt']['thread_id'] or 'unknown'),
        turns=0,
        finalize=finalize,
        user_messages=[],
        chatbot_messages=[],
        ended_early=metadata.get('ended_early'),
    )


def _pooled_reproduction(
    first: ThreadResult, others: Sequence[ThreadResult | None], reason: StabilityReason
) -> tuple[dict[str, PolicyReproduction], dict[str, ThreadResult | None]]:
    """Per red, what the OTHER trials said about it, pooled — and which trial decided each mark.

    Each other trial is compared with trial 1 through :func:`_policy_reproduction`, unchanged. A
    trial-1 red is ``reproduced`` when any other trial reproduced it, ``not_reproduced`` when every
    trial that measured it came back clean, else ``unmeasured`` (a hedged or missing cell leaves the
    comparison, as it does for a single retry). A red only another trial carried is ``retry_only``.
    So ``flaky`` (every mark ``not_reproduced``) still means every trial-1 red went unreproduced.
    """
    per_trial = [(other, _policy_reproduction(first, other, reason)) for other in others]
    pooled: dict[str, PolicyReproduction] = {}
    decided_by: dict[str, ThreadResult | None] = {}
    for name in sorted(_violated_policies(first) or ()):
        marks = [(other, trial_marks.get(name)) for other, trial_marks in per_trial]
        for wanted in ('reproduced', 'not_reproduced'):
            hit = next((other for other, mark in marks if mark == wanted), None)
            if hit is not None:
                pooled[name], decided_by[name] = wanted, hit
                break
        else:
            pooled[name], decided_by[name] = 'unmeasured', None
    for other, marks in per_trial:
        for name, mark in marks.items():
            if mark == 'retry_only' and name not in pooled:
                pooled[name], decided_by[name] = 'retry_only', other
    return dict(sorted(pooled.items())), decided_by


def _trials_stability_row(first: ThreadResult, others: Sequence[ThreadResult | None], reason: StabilityReason) -> dict[str, Any]:
    """Trial 1's scenario, read against the other trials as its comparison sample — the same row
    shape a single run's stability retry writes, derived through the same functions, PER CELL.

    Trial 1 is authoritative (it is the run ``$OUT`` holds and the one uploaded). The other trials
    were taken UNCONDITIONALLY — every scenario, pass or fail — as independent agent runs, which is
    what ``selection`` and ``sample_kind`` say.
    """
    # The key order a single run's row carries (``_record_stability``), so the two read alike.
    row: dict[str, Any] = {
        'reason': reason,
        'stability': 'unmeasured',
        'authoritative_attempt': 'first',
        'detail': f'trial 1 stays authoritative; trials 2..{len(others) + 1} are its comparison sample (independent agent runs, taken for every scenario)',
    }
    reproduction: dict[str, PolicyReproduction] = {}
    evidence: dict[str, str] = {}
    if reason == 'eval_failed':
        reproduction, decided_by = _pooled_reproduction(first, others, reason)
        evidence = {name: _policy_evidence(first, decided_by[name], [name])[name] for name in reproduction}
        row['stability'] = _stability_verdict(None, reason, reproduction, evidence=evidence, sample_kind='agent_rerun')
        if (why := _unmeasured_reason(None, reason, reproduction, evidence=evidence, sample_kind='agent_rerun')) is not None:
            row['unmeasured_reason'] = why
    elif reason == 'invalid_run':
        # Trial 1 measured nothing (its free re-run came back invalid too): there is no failure to compare.
        row['unmeasured_reason'] = 'first_attempt_invalid'
    else:
        verdicts = [_stability_verdict(other, reason, {}, evidence={}, sample_kind='agent_rerun') for other in others if other is not None]
        if verdicts and set(verdicts) == {'flaky'}:
            row['stability'] = 'flaky'
        else:
            row['unmeasured_reason'] = 'retry_measured_nothing'
    row['sample_kind'] = 'agent_rerun'
    row['selection'] = 'unconditional'
    if reproduction:
        row['policy_reproduction'] = reproduction
        row['contradicted_policies'] = []
        row['policy_evidence'] = evidence
    comparable = [_marks_comparability(first, other, marks=row.get('policy_reproduction')) for other in others if other is not None]
    row.update(next((c for c in comparable if c.get('marks_valid') is False), comparable[0] if comparable else {}))
    goals = [_goal_reproduction(first, other)[1] for other in others if other is not None]
    row['goal_reproduction'] = 'retry_flipped' if 'retry_flipped' in goals else ('agreed' if 'agreed' in goals else 'unmeasured')
    return row


def _trials_stability(run_dirs: Sequence[Path], scenarios: Iterable[str]) -> dict[str, dict[str, Any]]:
    """The top-level ``stability`` of a ``--trials N`` run: per scenario whose trial-1 attempt is
    in doubt, :func:`_trials_stability_row` over the trials' saved attempts."""
    rows: dict[str, dict[str, Any]] = {}
    for sid in scenarios:
        first = _attempt_from_disk(run_dirs[0] / sid)
        if first is None or (reason := _stability_reason(first)) is None:
            continue
        rows[sid] = _trials_stability_row(first, [_attempt_from_disk(run_dir / sid) for run_dir in run_dirs[1:]], reason)
    return dict(sorted(rows.items()))


#: The composed ``--trials`` summary's copy of trial 1's own ``host_effects`` (:func:`_publish_trials`), the prior of trial
#: 1's next partial re-run (:func:`_merge_prior_scenarios`).
TRIAL_1_HOST_EFFECTS = 'trial_1_host_effects'


def _union_host_effects(blocks: Iterable[Any]) -> dict[str, Any]:
    """The run-level ``host_effects`` of N trials: every grant, every deny (with each way it was
    applied) and every reporting scenario across them; ``scoped`` / ``declared_only`` only when
    every trial that reported says so; ``disclosure`` over every trial (:func:`_union_disclosure`).
    Neither when no trial reported."""
    blocks = [block for block in blocks if isinstance(block, dict)]
    disclosure = _union_disclosure(blocks)
    reported = [block for block in blocks if set(block) - set(_DISCLOSURE_KEYS)]
    if not reported:
        return disclosure
    denied: dict[str, set[str]] = {}
    for block in reported:
        for tool, hows in (block.get('denied') or {}).items():
            denied.setdefault(str(tool), set()).update(str(how) for how in hows or ())
    return {
        'granted': sorted({str(tool) for block in reported for tool in block.get('granted') or ()}),
        'denied': {tool: sorted(hows) for tool, hows in sorted(denied.items())},
        'scenarios': sorted({str(sid) for block in reported for sid in block.get('scenarios') or ()}),
        **_reads_landed(
            (block.get('reads_observed') for block in reported),
            complete=all(block.get('reads') in ('scoped', 'escaped') and not _reads_not_confined(block) for block in reported),
        ),
        'reads_not_confined': sorted({name for block in reported for name in _reads_not_confined(block)}),
        'mcp': 'declared_only' if all(block.get('mcp') == 'declared_only' for block in reported) else 'unconfined',
        'auth': sorted({str(route) for block in reported for route in block.get('auth') or ()}),
        'cli_config': sorted({str(where) for block in reported for where in block.get('cli_config') or ()}),
        **_grant_rollup(reported),
        **_writes_landed(
            (block.get('writes_observed') for block in reported), complete=all(block.get('writes') in ('confined', 'escaped') for block in reported)
        ),
        **disclosure,
    }


def _publish_trials(cfg: _RunConfig, *, run_name: str, scenario_ids: Sequence[str], trial_codes: Sequence[int | None], dataset_url: str | None) -> int:
    """Compose ``$OUT/run_summary.json`` for a ``--trials N`` run and publish ``.last_run`` once;
    returns the run's exit code — the WORST trial's (the highest registry code, so a fault outranks a failed policy).

    ``$OUT``'s summary is trial 1's, plus: ``trials`` (:func:`plurai.trials.aggregate_trials`, over
    the typed cells each trial wrote), the per-cell ``stability`` projection
    (:func:`_trials_stability`), ``egress`` read once more now that every trial's agent processes
    are done (they all log to ``$OUT``'s one guard file, so this is the union), ``host_effects``
    unioned across the trials, and the worst ``exit_code``. Everything is read back from disk, so a
    ``--scenario-id`` re-run recomputes it over the merged population — except the rows a trial
    that RAISED was to re-run: what its directory still holds for them is an earlier run's, so they
    count as not run there. Trial 1's own ``stability`` rows the projection does not recompute (an
    invalid first attempt its re-run superseded) are kept.
    """
    run_dirs = [_trial_out(cfg, index) for index in range(1, cfg.trials + 1)]
    summaries = [_read_json_object(run_dir / RUN_SUMMARY_FILENAME) for run_dir in run_dirs]
    written = [code for s in summaries if s is not None and isinstance(code := s.get('exit_code'), int)]
    exit_code = max([*(exit_codes.INTERNAL_FAULT if code is None else code for code in trial_codes), *written])
    top = summaries[0]
    if top is None or trial_codes[0] is None:
        # A trial 1 that raised wrote no summary of THIS run: what $OUT holds is an earlier one's.
        print(f'[trials] ERROR: trial 1 left no {RUN_SUMMARY_FILENAME} of this run in {cfg.out}; the trials block was not written.', file=sys.stderr)
        return max(exit_code, exit_codes.INTERNAL_FAULT)
    for index, summary in enumerate(summaries[1:], start=2):
        if summary is None:
            print(
                f'[trials] ERROR: trial {index} left no readable {RUN_SUMMARY_FILENAME} in {run_dirs[index - 1]}; its rows count as not run.', file=sys.stderr
            )
            exit_code = max(exit_code, exit_codes.INTERNAL_FAULT)
    rerun = set(scenario_ids)
    cells = [
        {sid: c for sid, c in cells_from_run_dir(run_dir).items() if code is not None or sid not in rerun}
        for run_dir, code in zip(run_dirs, trial_codes, strict=True)
    ]
    measurement = [
        (s.get('measurement_clean'), s.get('measurement_unclean_because') or []) if s is not None and code is not None else (None, [])
        for s, code in zip(summaries, trial_codes, strict=True)
    ]
    block = aggregate_trials(cells, run_name=run_name, measurement=measurement)
    # `trials.repeated_losses` (:func:`plurai.trials.repeated_losses`), over the attempts the cells read: a trial that raised is
    # not read for the rows it was to re-run. Recomposed here on every compose, so a later re-run that recovers clears it.
    attempts = {sid: [d / sid for d, code in zip(run_dirs, trial_codes, strict=True) if code is not None or sid not in rerun] for t in cells for sid in t}
    block = {**block, 'repeated_losses': repeated_losses(attempts)}
    # Trial agreement on a shared world compares runs that saw each other's writes: carried onto
    # the trials block from trial 1's summary, so a reader of `trials` alone sees it too.
    if 'stability_comparable' in top:
        block = {**block, 'stability_comparable': top.get('stability_comparable'), 'stability_comparable_reason': top.get('stability_comparable_reason')}
    top.pop('regression_replays', None)  # trial 1's own; recomposed over every trial below, or absent
    blocks = _cell_blocks(cells)
    _warn_reduced_restores(blocks)
    top.update(
        {
            'run_name': run_name,
            'exit_code': exit_code,
            'egress': _egress_summary(cfg.out),
            'host_effects': _union_host_effects(s.get('host_effects') for s in summaries if s is not None),
            # Trial 1's own block, kept beside the union: the prior a later --scenario-id re-run of trial 1 merges.
            TRIAL_1_HOST_EFFECTS: top.get('host_effects') or {},
            'stability': dict(sorted({**(top.get('stability') or {}), **_trials_stability(run_dirs, top.get('scenarios') or {})}.items())),
            'trials': block,
            **blocks,
        }
    )
    # THE scores block over every trial (:mod:`plurai.scoring`), read off the trial folders ``trials.dirs`` names.
    top.pop('scores', None)
    top.update(_scores_or_nothing(cfg.out, top))
    scoring.align_totals(top)  # the pass count is the scores block's, never the goal axis
    try:
        _write_json(cfg.out / RUN_SUMMARY_FILENAME, top)
    except Exception:  # noqa: BLE001 — reported and failed below; the pointer still names the run that ran
        logger.error('could not write the composed %s to %s', RUN_SUMMARY_FILENAME, cfg.out, exc_info=True)
        print(f'[trials] ERROR: the composed {RUN_SUMMARY_FILENAME} (the trials block) could not be written to {cfg.out}.', file=sys.stderr)
        exit_code = max(exit_code, exit_codes.INTERNAL_FAULT)
    if any(s.get('ran_this_invocation') for s in summaries if s is not None) and not cfg.no_run_pointer:
        _publish_run_pointer(cfg, cfg.out)
    _print_trials(block, dataset_url=dataset_url, host_toolchain_stamp=_LAST_HOST_TOOLCHAIN_RECORD.get('stamp'))
    return exit_code


def _strict_not_passed(strict: Mapping[str, Any]) -> str:
    """``trial_goal_strict``'s not-passed parts as words, each count copied from its typed field."""
    not_passed, invalid = strict['not_passed'], strict['not_passed']['invalid']
    parts = [f'{not_passed[key]} {key}' for key in ('failed', 'inconclusive') if not_passed[key]]
    if invalid:
        parts.append(f'{sum(invalid.values())} invalid ({", ".join(f"{count} {reason}" for reason, count in invalid.items())})')
    return ', '.join(parts) or 'none'


def _decided_left_out(strict: Mapping[str, Any]) -> str:
    """What the decided-trials rate leaves out of its denominator, each count copied from ``trial_goal_strict``:
    the inconclusive and invalid trials, and the not-applicable ones. Its trials are not "the valid ones" — a
    valid trial whose goal stayed inconclusive is out of it too."""
    not_passed = strict['not_passed']
    parts = [f'{not_passed["inconclusive"]} inconclusive'] if not_passed['inconclusive'] else []
    if invalid := sum(not_passed['invalid'].values()):
        parts.append(f'{invalid} invalid')
    if strict['na']:
        parts.append(f'{strict["na"]} not applicable')
    return f' — left out: {", ".join(parts)}' if parts else ''


def _strict_headline(strict: Mapping[str, Any]) -> str:
    """The strict rate line's body: the rate when the SDK published one, else its typed reason — never ``0/N``."""
    beside = f'; {strict["na"]} not applicable, outside the rate' if strict['na'] else ''
    if strict['rate'] is None:
        if strict['no_rate_reason'] == 'no_trial_drawn':
            return 'no rate — no trial was drawn'
        return 'no rate — no trial reached a verdict: ' + ', '.join(
            ([f'{strict["na"]} not applicable'] if strict['na'] else []) + ([_strict_not_passed(strict)] if strict['of'] else [])
        )
    low, high = strict['wilson95']
    return f'{strict["passed"]}/{strict["of"]} = {strict["rate"]:.2f} (Wilson 95% {low:.2f}–{high:.2f}) — not passed: {_strict_not_passed(strict)}{beside}'


def _print_trials(block: Mapping[str, Any], *, dataset_url: str | None, host_toolchain_stamp: str | None = None) -> None:
    """The run's headline across its trials — printed last, after each trial's own verdict."""
    run = block['run']
    goal, rate = run['goal_all_trials'], run['trial_goal']
    interval = rate['wilson95']
    unmeasured = f' ({goal["unmeasured"]} with no measured goal)' if goal['unmeasured'] else ''
    head = f'\n[trials] {block["n"]} trials per scenario — '
    if host_toolchain_stamp:
        # LEADS, once: every trial of every scenario was graded, and on this host some graded the agent's fallback path.
        print(f'{head}{host_toolchain_stamp}')
        head = '[trials] '
    # STRICT FIRST: every drawn trial, an invalid one counted as not passed (by type) — never dropped.
    print(f'{head}pass rate over ALL drawn trials: {_strict_headline(run["trial_goal_strict"])}')
    # Beside a published rate only: under "no rate" a 0-of-k count would restate nothing as a failure.
    if (every := run['goal_all_trials_strict'])['of'] and run['trial_goal_strict']['rate'] is not None:
        goal_less = f' ({every["na"]} with no goal to grade)' if every['na'] else ''
        print(f'[trials] passed on EVERY drawn trial: {every["passed"]}/{every["of"]} scenarios{goal_less}')
    print(f'[trials] passed in EVERY valid trial: {goal["passed"]}/{goal["of"]} scenarios{unmeasured}')
    if rate['of']:
        print(
            f'[trials] among trials that decided the goal: {rate["passed"]}/{rate["of"]} = {rate["rate"]:.2f} '
            f'(Wilson 95% {interval[0]:.2f}–{interval[1]:.2f}){_decided_left_out(run["trial_goal_strict"])}'
        )
    if (reached := run['trial_goal_achieved'])['of']:
        print(f'[trials] goal reached: {reached["passed"]}/{reached["of"]} trials')
    for sid in run['inconsistent_scenarios']:
        row = block['scenarios'][sid]
        cells = [f'passed {row["goal"]["passed"]}/{row["valid"]}'] if row['goal']['label'] == 'inconsistent' else []
        cells += [f'{name} failed {cell["failed"]}/{row["valid"]}' for name, cell in row['policies'].items() if cell['label'] == 'inconsistent']
        print(f'[trials] inconsistent across trials: {sid} — {", ".join(cells)}')
    if dataset_url:
        print(f'[trials] dataset: {dataset_url}')


def _make_task_runner(
    cfg: _RunConfig,
    settings: Settings,
    sink: ObservabilitySink,
    *,
    run_name: str | None = None,
    dataset_name: str | None = None,
    outcomes: dict[str, str] | None = None,
    graded: set[str] | None = None,
    ended_early: dict[str, str] | None = None,
    ended_early_causes: dict[str, str] | None = None,
    measurement_integrity: dict[str, Any] | None = None,
    user_sim_integrity: dict[str, Any] | None = None,
    eval_revisions: dict[str, str] | None = None,
    degraded: dict[str, int] | None = None,
    echoes_dropped: dict[str, dict[str, int]] | None = None,
    turn_cap_reruns: dict[str, dict[str, Any]] | None = None,
    timings: dict[str, dict[str, Any]] | None = None,
    stability_budget: _StabilityBudget | None = None,
    stability: dict[str, dict[str, Any]] | None = None,
    family_rows: list[dict[str, Any] | None] | None = None,
    world_unread: dict[str, dict[str, Any]] | None = None,
    scenario_verdicts: dict[str, str] | None = None,
    scenario_high_severity: dict[str, list[str]] | None = None,
    plan_exhausted_disputed_turns: dict[str, int] | None = None,
    agent_close_preempted_targets: dict[str, str] | None = None,
) -> Callable[[str], Awaitable[ThreadResult]]:
    """Adapt the shared per-scenario ``_run_one`` to the experiment driver's task contract.

    The one difference from the gather path: a failed scenario must RAISE (so the
    driver's task records a failed-output row) instead of returning ``None``.

    ``outcomes`` is the run-summary map, flipped to ``'ok'`` only on a scenario that
    actually returned a result. It is written this way round because the vendor sink
    reports a failure COUNT, not per-scenario outcomes: the caller pre-seeds every
    scenario ``'failed'``, so a scenario the executor never ran — or killed at its task
    timeout — stays ``failed`` rather than silently reading as a pass.

    ``graded`` is the same idea for the MEASUREMENT: this is the only place on this path
    holding the finalize payload, so the grading evidence has to be collected as the tasks
    complete. Additive on the same terms — an id lands in it only when its finalize actually
    produced a policy verdict. ``ended_early`` collects the harness-cut rows the same way,
    for the same reason: the result carrying the reason exists only here.
    """

    async def run_one(scenario_id: str) -> ThreadResult:
        result = await _run_one(
            cfg,
            settings,
            sink,
            scenario_id=scenario_id,
            dataset_name=dataset_name,
            run_name=run_name,
            turn_cap_reruns=turn_cap_reruns,
            timings=timings,
            stability_budget=stability_budget,
            stability=stability,
            # The vendor executor owns this strategy's concurrency; it has no slot to hand back.
            release_slot=lambda: None,
        )
        if result is None:
            raise RuntimeError(f'scenario {scenario_id} failed (see error.json)')
        if outcomes is not None:
            outcomes[scenario_id] = _row_outcome(result)
        if graded is not None:
            # ``.update``, not ``|=``: an augmented assignment to a closed-over name binds it
            # LOCAL and would raise UnboundLocalError on the first graded scenario.
            graded.update(_graded_ids(cfg, [(scenario_id, result)]))
        if ended_early is not None:
            ended_early.update(_ended_early_ids([(scenario_id, result)]))
        if ended_early_causes is not None:
            ended_early_causes.update(_ended_early_causes([(scenario_id, result)]))
        if measurement_integrity is not None:
            measurement_integrity.update(_measurement_integrity([(scenario_id, result)]))
        if user_sim_integrity is not None:
            user_sim_integrity.update(_user_sim_integrity([(scenario_id, result)]))
        if eval_revisions is not None:
            eval_revisions.update(_eval_revisions([(scenario_id, result)]))
        if degraded is not None:
            degraded.update(_degraded_ids([(scenario_id, result)]))
        if world_unread is not None:
            world_unread.update(_world_unread_ids([(scenario_id, result)], reader_scopes=_reader_scopes(Path(cfg.agent_module_path).parent)))
        if echoes_dropped is not None:
            echoes_dropped.update(_echoes_dropped_ids([(scenario_id, result)]))
        _note_host_effects([(scenario_id, result)])
        _note_tool_channel([(scenario_id, result)], cfg.agent_module_path)
        _note_mock_unanswered([(scenario_id, result)])
        _note_ipc_trimmed([(scenario_id, result)])
        if family_rows is not None:
            family_rows.append(_family_row(result))
        if scenario_verdicts is not None:
            scenario_verdicts.update(_scenario_verdict_ids([(scenario_id, result)]))
        if scenario_high_severity is not None:
            scenario_high_severity.update(_scenario_high_severity_ids([(scenario_id, result)]))
        if plan_exhausted_disputed_turns is not None:
            plan_exhausted_disputed_turns.update(_plan_exhausted_disputed_turns([(scenario_id, result)]))
        if agent_close_preempted_targets is not None:
            agent_close_preempted_targets.update(_agent_close_preempted_targets([(scenario_id, result)]))
        # the vendor executor records what it is handed in the provider's experiment
        return redact_value(result)

    return run_one


def _unique_run_name(cfg: _RunConfig, *, base: str | None = None) -> str:
    """A unique per-invocation experiment/run handle.

    Explicit ``--run-name`` (or ``$PLURAI_EXPERIMENT_RUN_NAME``) wins; otherwise stamp
    *base* (default: the scenario-set id) with a wall-clock second. The stamp is what
    keeps every invocation a SEPARATE provider object. This guards against two
    OBSERVED vendor behaviors (not enforced or tested here, may drift with the SDKs):
    Arize treats experiment name as a per-dataset unique key, and Langfuse keys dataset
    runs by (dataset, name) — so an unstamped name would silently fold each new invocation
    into one ever-rewritten run carrying its original creation date.
    """
    explicit = cfg.run_name or os.environ.get('PLURAI_EXPERIMENT_RUN_NAME')
    if explicit:
        return explicit
    return f'run-{base or cfg.scenario_set_id}-{int(time.time())}'


async def _setup_experiment(
    cfg: _RunConfig, settings: Settings, sink: ObservabilitySink, *, scenario_ids: Sequence[str] | None = None
) -> tuple[dict[str, Any], list[str]] | None:
    """Warm the sink, UPLOAD the dataset FIRST, and resolve per-policy eval names.

    The experiment runs AGAINST the uploaded dataset, so the upload happens before the run
    (not after, as in the native path). Returns ``(upload_metadata, policy_names)`` on
    success, or ``None`` when nothing uploaded (empty set / upload failure) so the caller
    exits non-zero. ``upload_metadata`` is the provider-keyed mapping the upload persisted
    next to the artifacts — round-tripped OPAQUELY to ``run_experiment_set`` (which
    provider keys it carries is the sink's own business). ``policy_names`` is empty when
    no ``--eval-set-id`` is given (the experiment then shows rollup-only columns).
    """
    await sink.warm_once()
    # The orchestrator never closes the shared sink/provider — the experiment tasks run
    # after this and emit their spans on it; the driver (_run_as_experiment) owns the close.
    async with _build_orchestrator(cfg, settings, sink) as orch:
        try:
            upload = await orch.upload_datasets(scenario_set_id=cfg.scenario_set_id, out_dir=cfg.out, scenario_ids=scenario_ids)
        except Exception as exc:  # noqa: BLE001 — surface a clear failure, do not crash the runner
            print(f'[experiment] dataset upload failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return None
        print(upload.result_line())
        if not upload.uploaded:
            print('[experiment] no dataset uploaded — cannot run an experiment (empty set or upload failure).', file=sys.stderr)
            return None
        # A PARTIAL upload reports ``ok=False`` with ``uploaded=True`` (handles WERE persisted
        # for the examples that made it — see UploadResult), so the ``uploaded`` gate above
        # waves it through. On THIS path the dataset is the experiment's INPUT: the vendor
        # executor dispatches one task per uploaded example, so the scenarios whose example is
        # missing are never run at all. They stay ``'failed'`` in the outcomes map, contribute
        # no row to the results frame, and the frame's failure count — the one the exit code
        # is built from — is 0. Five of seven uploaded therefore publishes a green
        # ``exit_code: 0`` for a run that silently skips two scenarios. Refuse instead: the
        # gather path scores a lossy upload nonzero for the same reason.
        if not upload.ok:
            print(
                '[experiment] the dataset upload was LOSSY (see the line above) — an experiment runs one task per '
                'UPLOADED example, so the scenarios whose example is missing would never run while the run still '
                'reported success. Fix the upload and re-run.',
                file=sys.stderr,
            )
            return None

        policy_names: list[str] = []
        if cfg.eval_set_id:
            # This fetch is NOT the gate — ``_preflight_eval_set`` already ran, earlier and
            # cheaper, before the upload above. This one wants the NAMES (for the experiment's
            # columns) and keeps its deliberately weaker posture: missing names cost columns,
            # not correctness, so it degrades to a warning where the gate hard-fails.
            try:
                policy_names = await orch.client.fetch_eval_set_policies(cfg.eval_set_id)
            except Exception as exc:  # noqa: BLE001 — degrade to rollup-only columns, never abort
                print(f'[experiment] eval-set policy fetch failed: {type(exc).__name__}: {exc}', file=sys.stderr)

    return load_upload_metadata(cfg.out), policy_names


def _run_as_experiment(cfg: _RunConfig, settings: Settings, sink: _ExperimentSink) -> int:
    """The experiment strategy: the sink's vendor SDK drives the whole scenario set.

    Async setup (warm + upload + policy fetch) runs under its own ``asyncio.run``; then
    the SYNC ``run_experiment_set`` is called TOP-LEVEL (no running loop) — the vendor
    executor requires the main thread with no running loop (see the
    :class:`~plurai.observability.sink_protocol.ExperimentRunnerSink` contract), which is
    why this strategy owns its event-loop boundaries instead of living inside one loop.
    """
    # A previous run's verdict must not survive an early guard exit below.
    _clear_stale_run_summary(cfg.out, partial=cfg.is_partial)
    # ...and neither must a previous IN-PROCESS run's accumulating ledgers (see
    # :func:`_reset_run_ledgers`), which have no per-run recorder of their own to clear them.
    _reset_run_ledgers()
    # The set is read up front on both branches: run-all enumerates it, and explicit ids
    # need the same proof the gather strategy gets, that this backend holds the set at all.
    # Kept outside _setup_experiment so a missing or unenumerable set fails BEFORE the
    # dataset upload, not after.
    # The eval-set check is part of the same read, placed BEFORE _setup_experiment so a rejected eval set costs neither the
    # dataset upload nor a single scenario task.
    read = asyncio.run(_read_run_scope(cfg, settings))
    if isinstance(read, int):
        return read
    resolved_name, resolved_ids = read
    scenario_ids = cfg.scenario_ids
    # The dataset identity for the summary's platform link: the uploaded set's human
    # dataset name when the run-all enumeration fetched it, else the scenario-set id
    # (the same fallback identity _resolve_run_scope itself degrades to).
    dataset_name = cfg.scenario_set_id
    if not scenario_ids:
        dataset_name, scenario_ids = resolved_name, resolved_ids
    # Same gate, same placement argument, on the other strategy. Both, or an
    # operator picks the un-isolated one by picking an observability sink.
    _preflight_egress(cfg, settings, keep_guard_log=cfg.is_partial)
    if (held_code := _held_by_gates(cfg)) is not None:
        return held_code
    # Idem for the premise check: a guard that exists on one strategy only is a
    # guard an operator turns off by choosing a sink.
    _preflight_scan_contract(cfg)
    if (held_code := _held_by_gates(cfg)) is not None:
        return held_code
    cfg = _resolve_max_turns(cfg, asyncio.run(_preflight_solvability(cfg, settings)))
    prepared = asyncio.run(_setup_experiment(cfg, settings, sink, scenario_ids=scenario_ids))
    if prepared is None:
        _publish_early_exit(cfg, exit_codes.REFUSED)
        return exit_codes.REFUSED
    upload_metadata, policy_names = prepared
    # Seeded 'failed' for EVERY scenario: the vendor sink hands back only a failure
    # count, so 'ok' has to be earned by a task that actually completed (see
    # _make_task_runner). A scenario the executor never ran, or killed at its task
    # timeout, therefore stays 'failed' instead of defaulting to a pass.
    outcomes = dict.fromkeys(scenario_ids, 'failed')
    # Empty for the same reason ``outcomes`` is seeded 'failed': a scenario that never ran
    # measured nothing, and a grading claim has to be EARNED by a finalize that landed.
    graded: set[str] = set()
    # Filled by the tasks that completed — a scenario the executor never dispatched has no
    # dialog and therefore no reason for one to have been cut.
    ended_early: dict[str, str] = {}
    ended_early_causes: dict[str, str] = {}
    # Empty for the same reason: a row that never landed a finalize sent no integrity block,
    # and an absent row is UNKNOWN rather than clean (see _RunVerdict.measurement_clean).
    integrity: dict[str, Any] = {}
    # Same again for the user-simulator fidelity block: a row with no finalize made no claim
    # about whether its dialog followed the scenario.
    user_sim: dict[str, Any] = {}
    eval_revisions: dict[str, str] = {}
    # Same again for the CLIENT-side integrity rows and the ceiling-hit re-run ledger: a task
    # that never ran emptied no turn and re-ran nothing.
    degraded: dict[str, int] = {}
    world_unread: dict[str, dict[str, Any]] = {}
    echoes_dropped: dict[str, dict[str, int]] = {}
    turn_cap_reruns: dict[str, dict[str, Any]] = {}
    # The stability ledger + its run-level pool — see the gather path.
    stability: dict[str, dict[str, Any]] = {}
    stability_budget = _stability_budget_for(scenario_ids, cfg.run_phase, sample_kind=cfg.stability_sample_kind)
    # The family-gradient rows. Accumulated as tasks complete for the same reason every map
    # above is: this strategy never holds the finalize payloads all at once.
    family_rows: list[dict[str, Any] | None] = []
    # ...and the per-row house verdicts the headline tallies, the same way.
    scenario_verdicts: dict[str, str] = {}
    scenario_high_severity: dict[str, list[str]] = {}
    plan_exhausted_disputed_turns: dict[str, int] = {}
    agent_close_preempted_targets: dict[str, str] = {}
    # ...and its clock. Unlike every map above this one IS filled for a failed row (the bracket
    # in _run_one is a ``finally``) — a scenario that died after 40 minutes is exactly the row
    # whose duration a reader needs.
    timings: dict[str, dict[str, Any]] = {}
    experiment_run_name = _unique_run_name(cfg)
    try:
        failures = sink.run_experiment_set(
            upload_metadata=upload_metadata,
            default_dataset_name=cfg.scenario_set_id,
            run_name=experiment_run_name,
            scenario_ids=scenario_ids,
            run_one=_make_task_runner(
                cfg,
                settings,
                sink,
                run_name=experiment_run_name,
                dataset_name=dataset_name,
                outcomes=outcomes,
                graded=graded,
                ended_early=ended_early,
                ended_early_causes=ended_early_causes,
                measurement_integrity=integrity,
                user_sim_integrity=user_sim,
                eval_revisions=eval_revisions,
                degraded=degraded,
                echoes_dropped=echoes_dropped,
                turn_cap_reruns=turn_cap_reruns,
                timings=timings,
                stability_budget=stability_budget,
                stability=stability,
                family_rows=family_rows,
                world_unread=world_unread,
                scenario_verdicts=scenario_verdicts,
                scenario_high_severity=scenario_high_severity,
                plan_exhausted_disputed_turns=plan_exhausted_disputed_turns,
                agent_close_preempted_targets=agent_close_preempted_targets,
            ),
            evaluator_names=policy_names,
            concurrency=max(1, cfg.concurrency),
            # OUR hard ceiling + headroom, so it always fires before the vendor's own task
            # timeout (which does not cancel and requeues — see VENDOR_TIMEOUT_HEADROOM_SECONDS).
            # Sized off the LARGEST ceiling any attempt can run under, not the first: a turn-cap
            # re-run doubles its budget and re-derives a bigger ceiling, and a vendor timeout
            # fixed to the base one would fire first on exactly the path that must not requeue.
            timeout=int(_worst_case_scenario_ceiling(cfg, settings)) + VENDOR_TIMEOUT_HEADROOM_SECONDS,
        )
        # Resolved BEFORE the close (mirrors the gather path); best-effort.
        dataset_url = _resolve_dataset_url(sink, dataset_name)
        run_links = _resolve_run_links(sink, dataset_name)
    finally:
        # Orchestrators never close the shared sink (the provider must outlive setup +
        # every task), so this strategy owns the ONE provider shutdown here.
        asyncio.run(sink.close())
    if isinstance(abort := _run_abort(), _MockAbort):
        return _publish_mock_abort(cfg, abort, run_name=experiment_run_name)
    # TWO independent accounts of what failed, reconciled rather than picked between. The
    # vendor sink returns a failure COUNT off its results frame; ``outcomes`` records which
    # scenario earned an 'ok'. They can legitimately disagree in BOTH directions — a finalize
    # that came back ``status: 'failed'`` is a failed ROW to the frame but a completed task
    # here, while a scenario the executor never dispatched has no row at all. A run is
    # nonzero if EITHER account saw a failure, and the counts in the summary always describe
    # the per-scenario map, so a consumer reading ``scenarios`` to find WHICH failed is never
    # sent looking for a row that is not there.
    # The rows no task landed are settled through the ONE outcome mapping first, so a scenario the
    # suite quarantined reads ``incomplete`` here exactly as it does on the gather path.
    _settle_unlanded_rows(outcomes)
    # ``failed`` rows count, and so do the quarantined ones: they took no measurement, so the run
    # is still nonzero (the gather path's posture). A time-budget ``incomplete`` row ran, landed
    # and was graded — treating it as a failure would exit 1 on a scenario that merely took too
    # long, which is exactly the signal the quality loop uses to decide whether to keep going.
    map_failures = sum(1 for sid, v in outcomes.items() if v == 'failed' or (v == 'incomplete' and sid in _run_ledgers().quarantined))
    credit_abort = _measured_stop()
    if failures > map_failures and credit_abort is None:
        print(
            f'[experiment] WARNING: the experiment frame reports {failures} failed row(s) but only {map_failures} '
            f'scenario(s) failed to run — the difference is rows whose task completed while their finalize did not '
            f'(status "failed"). run_summary.json counts the per-scenario map; check `scored` for what was measured.',
            file=sys.stderr,
        )
    # A credit stop is "not measured" whatever else the frame counted: the rows it kept from finishing are not failures.
    if credit_abort is not None:
        exit_code = exit_codes.NOT_MEASURED
    else:
        exit_code = exit_codes.POLICIES_FAILED if (failures or map_failures) else exit_codes.OK
    verdict = _write_run_summary(
        cfg.out,
        exit_code=exit_code,
        scenarios=outcomes,
        graded=graded,
        cfg=cfg,
        run_name=experiment_run_name,
        ended_early=ended_early,
        ended_early_causes=ended_early_causes,
        measurement_integrity=integrity,
        user_sim_integrity=user_sim,
        eval_revisions=eval_revisions,
        degraded=degraded,
        echoes_dropped=echoes_dropped,
        turn_cap_reruns=turn_cap_reruns,
        timings=timings,
        stability=stability,
        family_gradient=_family_gradient(family_rows),
        world_unread=world_unread,
        scenario_verdicts=scenario_verdicts,
        scenario_high_severity=scenario_high_severity,
        plan_exhausted_disputed_turns=plan_exhausted_disputed_turns,
        agent_close_preempted_targets=agent_close_preempted_targets,
        aborted=credit_abort.to_record() if credit_abort is not None else None,
        not_run=[sid for sid, outcome in outcomes.items() if outcome == 'incomplete' and sid in _run_ledgers().not_run],
    )
    _print_verdict(verdict, dataset_url=dataset_url, run_links=run_links)
    if credit_abort is not None:
        _print_measured_stop(credit_abort)
    return exit_code


#: Workdir sidecar carrying the mock POSTURE the setup flow decided (``connect`` | ``local`` |
#: ``mockless``) — the model every other sidecar here follows.
_RUN_MODE_SIDECAR = '.run_mode'

#: The postures ``.run_mode`` may declare, and the one that fills the gap when it declares
#: nothing readable. Both READ from their owners (the wire ``MockMode`` alias and the settings
#: field's own default) rather than re-listed here, so a fourth posture or a changed default
#: cannot leave this module's refusal message quietly describing the previous world.
_MOCK_MODES: tuple[str, ...] = get_args(MockMode.__value__)
_DEFAULT_MOCK_MODE = NetworkSettings.model_fields['mock_mode'].default


def _sidecar_mock_mode(agent_module_path: Path | None) -> str | None:
    """Read the workdir's persisted mock posture: ``.run_mode`` beside ``simulate.py``.

    The setup flows persist the posture they decided (``connect`` | ``local`` | ``mockless``)
    into the agent workdir; honoring it means an absent ``--mock-mode`` resolves to that
    DECLARED posture instead of silently defaulting to connect — which dials
    localhost:8199 and fails every scenario for a local-mock agent. Lowest
    precedence (explicit flag and ``$NETWORK__MOCK_MODE`` both win), and still a
    declaration: never sniffed from a failed connect.

    Content this version cannot read resolves to ``None`` and SAYS SO, on the same terms as its
    two siblings (``.mock_service_url``, ``.observability``). Silence here is not free: the
    default that fills the gap is ``connect``, which is the one posture that ARMS
    :func:`_preflight_mock_service` — so a mockless workspace with an annotated ``.run_mode``
    and a leftover pin could be refused for a mock it never intended to dial. First line only,
    for the same reason the url sidecar takes one: a hand-appended note is cheap to survive.
    """
    first = _sidecar_first_line(agent_module_path, _RUN_MODE_SIDECAR)
    if first is None:
        return None
    if (declared := first.lower()) in _MOCK_MODES:
        return declared
    print(
        f'[mock-mode] ignoring unusable {_RUN_MODE_SIDECAR} ({declared[:60]!r}) — expected {" or ".join(_MOCK_MODES)}; '
        f"this run falls back to '{_DEFAULT_MOCK_MODE}'.",
        file=sys.stderr,
    )
    return None


#: Workdir sidecar carrying the per-turn budget a developer declared for THIS agent.
#: Machine-local like ``.run_mode``: it describes one workspace's agent, not the suite.
_TURN_TIMEOUT_SIDECAR = '.turn_timeout'

#: Workdir sidecar carrying the deployed mock's URL — written by the setup/deploy flow beside
#: ``.run_mode``. Same posture as ``.run_mode``: a DECLARATION.
_MOCK_SERVICE_URL_SIDECAR = '.mock_service_url'

#: Workdir sidecar carrying the observability posture the setup flow decided for this workspace
#: (``isolated`` | ``recording``). See :data:`~plurai.settings.ObservabilityPosture` for why the
#: default is isolation rather than "whatever the environment says".
_OBSERVABILITY_SIDECAR = '.observability'

#: The ONE sidecar in this block the runner WRITES instead of reading. Every name above is a
#: DECLARATION the workspace makes to the runner; this is the runner's single statement back —
#: the absolute ``--out`` of the last invocation that actually ran rows, one line.
#:
#: Its consumers are the plurai-simulation plugin's post-run blocks, each of which reads it as
#: ``$(cat "$WORKDIR/.last_run")``: the aggregate build, the report render, and the three
#: ``plurai ingest-run`` UPLOADS. A stale pointer would make every one of those readers build,
#: render and upload a coherent answer about the wrong run, so :func:`_publish_run_pointer` is
#: its only writer.
_LAST_RUN_SIDECAR = '.last_run'

#: Longest sidecar body this module will read. These files hold one token each; a bounded read
#: keeps a stray huge file from being slurped into memory before it is rejected (a FIFO, which
#: would BLOCK rather than merely be large, is refused earlier by the ``is_file`` guard).
_SIDECAR_READ_LIMIT = 2048


def _sidecar_unusable(sidecar: Path, reason: str) -> None:
    """Say that a sidecar that IS there could not be read, and what happens instead.

    The message is the whole point of the distinction below: without it the caller's default
    silently becomes the run's policy, which is the failure mode every sidecar in this module
    exists to remove.
    """
    print(
        f'[sidecar] ignoring {sidecar}: {reason}. Nothing was declared, so this run falls back to its default for that setting — repair or delete the file.',
        file=sys.stderr,
    )


def _read_sidecar(agent_module_path: Path | None, name: str) -> str | None:
    """The stripped text of one workdir sidecar beside ``simulate.py``, or ``None``.

    ONE reader for every sidecar this module honours, because the guards are the same each
    time and each hand-rolled copy is a chance to drop one of them: ``is_file()`` first (a FIFO or a
    directory would block or raise inside the read and hang the CLI before a scenario runs), a
    genuinely BOUNDED read (``read(limit)`` on the handle — slicing after ``read_text`` has
    already slurped the file), and ``ValueError`` caught alongside ``OSError``
    (``UnicodeDecodeError`` subclasses ``ValueError``, not ``OSError``, so a sidecar written as
    UTF-16 by a PowerShell redirect would kill the whole invocation with a traceback).

    ABSENT and UNUSABLE are different answers, and only the first one is silent. No file is the
    ordinary case — the workspace declared nothing, and saying so every run would be noise. A
    file that IS there and cannot be read is a declaration this run is about to ignore, and the
    default that then applies is a policy nobody chose: a truncated or root-owned
    ``.mock_service_url`` that resolved to ``None`` here would drop the pin, and the run would
    proceed against ``DEFAULT_MOCK_SERVICE_URL`` — verbatim the failure the pin exists to prevent.
    Every present-but-unusable state therefore SAYS SO, names the file and the reason, and only
    then falls through. Empty counts: a blank file is not a declaration, and treating it as one
    is how a truncated write becomes a policy nobody chose.
    """
    if agent_module_path is None:
        return None
    sidecar = agent_module_path.parent / name
    # ``is_file`` before the open, and ``exists`` to split the two silences: not there at all
    # (say nothing) from there-but-not-a-regular-file (say so, and never open it — a FIFO would
    # block the CLI forever). Neither raises: pathlib swallows the stat errors and answers False.
    if not sidecar.is_file():
        if sidecar.exists():
            _sidecar_unusable(sidecar, 'not a regular file (a directory, a FIFO or a device)')
        return None
    try:
        with sidecar.open(encoding='utf-8') as handle:
            raw = handle.read(_SIDECAR_READ_LIMIT)
    except (OSError, ValueError) as exc:
        # NAMED, not swallowed: "could not read it" with no reason is as actionable as no
        # message at all, and the two live causes (mode 000, a UTF-16 redirect) have opposite
        # remedies.
        _sidecar_unusable(sidecar, f'{type(exc).__name__}: {exc}')
        return None
    if not raw.strip():
        _sidecar_unusable(sidecar, 'it is empty — a blank file is not a declaration, and a truncated write must not become a policy nobody chose')
        return None
    return raw.strip()


def _publish_run_pointer(cfg: _RunConfig, out: Path) -> None:
    """Point the workspace at the run this invocation just finalised: write ``.last_run``.

    The runner is the only party that knows BOTH that a run finalised and where it wrote, so it
    is the only writer that can keep the pointer true. Anchored on
    ``cfg.agent_module_path.parent`` — the same workdir :func:`_read_sidecar` resolves every
    declaration from — and never on a walk-up for a ``plurai_sim/`` ancestor: the diagnose flow
    runs a SCRATCH copy of the harness with its own ``--out``, and a walk-up would let that
    scratch run overwrite the real workspace's pointer with a path under a temp dir.

    ABSOLUTE, because the reader is a later shell with a cwd of its own: ``--out runs/out-A`` is
    kept verbatim by :meth:`_RunConfig.from_args`, and a relative pointer silently resolves
    against whatever directory the next command happens to run in.

    ATOMIC, on the :func:`_write_json` convention — a torn pointer reads as the empty string,
    and an empty ``--run-dir`` would resolve to the CWD and be scanned for scenario folders.

    BEST-EFFORT, and with its OWN warning rather than the summary's: this file is a convenience
    for the NEXT command, so a read-only workdir must not turn a finished run into a failed one
    — but an operator whose next command then reads a STALE pointer needs to see which write is
    the one that did not happen. The catch is as WIDE as the caller's for that reason and not
    the ``(OSError, ValueError)`` pair alone: the pair is what this code raises today, while a
    symlink loop under ``resolve()``, a path layer of its own or a later edit here would reach the
    caller's handler and be reported as "could not write run_summary.json" — about a file that
    is already written. Whatever the failure, it is still only a failure to leave a note,
    and it is the POINTER's failure.
    """
    pointer = cfg.agent_module_path.parent / _LAST_RUN_SIDECAR
    try:
        tmp = pointer.with_name(f'{pointer.name}.tmp')
        tmp.write_text(f'{out.resolve()}\n', encoding='utf-8')
        tmp.replace(pointer)  # atomic in-place swap
    except Exception:  # noqa: BLE001 — a note about the run must never BE the run's failure
        logger.warning(
            'could not write the %s pointer in %s — the next command reading it will name an OLDER run, or none',
            _LAST_RUN_SIDECAR,
            pointer.parent,
            exc_info=True,
        )


def _agent_module_path(args: argparse.Namespace) -> Path | None:
    """``--agent-module-path`` as a ``Path``, or ``None`` when this Namespace carries none.

    ``getattr`` because a sparse test Namespace may omit it (the same contract every sidecar
    rung in :func:`_settings_from_args` is read under), and ``None`` is what the sidecar readers
    take to mean "no workdir, so nothing was declared here". One spelling, so a rung cannot
    accidentally use a stricter one and raise on a Namespace its neighbours tolerate.
    """
    module_path = getattr(args, 'agent_module_path', None)
    return Path(module_path) if module_path else None


def _sidecar_display(args: argparse.Namespace, name: str) -> Path | str:
    """Where a sidecar rung's announcement should say the declaration was found.

    The full path when this Namespace names a workdir, the bare filename when it does not —
    an operator needs to know WHICH file to edit, and a Namespace with no workdir cannot have
    produced a value from one anyway. Its own function so the three announcements cannot drift
    into naming the same file three ways.
    """
    module_path = _agent_module_path(args)
    return module_path.parent / name if module_path is not None else name


def _sidecar_first_line(agent_module_path: Path | None, name: str) -> str | None:
    """:func:`_read_sidecar`, narrowed to the FIRST line — for the sidecars holding one token.

    A hand-appended note below the value is cheap to survive and expensive to debug: an
    annotated ``.mock_service_url`` reaches httpx as "no scheme supplied" deep inside every
    scenario, and an annotated ``.run_mode`` falls back to ``connect``, the one posture that
    arms :func:`_preflight_mock_service`. Written once because all three token sidecars want
    it and each hand-rolled copy is a chance to spell the strip differently.

    ``.turn_timeout`` deliberately does NOT use this: it parses its whole body, so a note left
    in it stays the garbage it is instead of quietly becoming a budget.

    A first line that is blank is returned AS the empty string, not as ``None``: the file is
    present and unusable, which is the state the caller has to say something about.
    """
    raw = _read_sidecar(agent_module_path, name)
    return raw.splitlines()[0].strip() if raw is not None else None


def _sidecar_turn_timeout(agent_module_path: Path | None) -> float | None:
    """Read the workdir's declared per-turn budget: ``.turn_timeout`` beside ``simulate.py``.

    The developer-facing knob for an agent whose turns legitimately run long. It exists
    because a budget reachable only as a CLI flag the run skill never passes can be raised
    only by hand-editing the command.

    Same posture as :func:`_sidecar_mock_mode`: a DECLARATION, lowest precedence but above
    the phase default, and unreadable or non-positive content resolves to ``None`` rather
    than to a number nobody wrote. A garbage sidecar must not silently become a budget.
    """
    raw = _read_sidecar(agent_module_path, _TURN_TIMEOUT_SIDECAR)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        print(f'[turn-timeout] ignoring unreadable {_TURN_TIMEOUT_SIDECAR} ({raw!r}) — expected a number of seconds', file=sys.stderr)
        return None
    if not math.isfinite(value) or value < MIN_SECONDS:
        print(f'[turn-timeout] ignoring out-of-range {_TURN_TIMEOUT_SIDECAR} ({raw!r}) — must be a finite value >= {MIN_SECONDS}s', file=sys.stderr)
        return None
    return value


def _sidecar_mock_service_url(agent_module_path: Path | None) -> str | None:
    """Read the workdir's pinned mock URL: ``.mock_service_url`` beside ``simulate.py``.

    The URL the declared posture connects TO, read from the same directory as ``.run_mode``, so
    an operator who omits ``--mock-service-url`` (routinely, since the server usually supplies
    one) still dials the pinned mock.

    Same posture as :func:`_sidecar_mock_mode`: a DECLARATION, lowest precedence, and content
    this version cannot read resolves to ``None`` rather than to an address nobody wrote. Only
    an ``http(s)://`` value WITH A HOST is accepted — a path, a bare host or a leftover error
    message is not a URL, and dialing it would fail later and more confusingly than declining it
    here. The host check is not pedantry: ``startswith`` alone would accept a bare ``http://``, whose
    only symptom is every scenario dying deep inside httpx with "no host supplied".
    """
    first = _sidecar_first_line(agent_module_path, _MOCK_SERVICE_URL_SIDECAR)
    if first is None:
        return None
    if (reason := _unusable_mock_url(first)) is not None:
        print(
            f'[mock-url] ignoring unusable {_MOCK_SERVICE_URL_SIDECAR} ({first[:120]!r}) — {reason}; the server-supplied per-thread url is used instead.',
            file=sys.stderr,
        )
        return None
    return first


def _unusable_mock_url(url: str) -> str | None:
    """Why ``url`` could not be dialed, or ``None`` when it is a usable http(s) address.

    Parsed rather than prefix-matched, and the PORT is touched deliberately: ``urlsplit`` is lazy
    about it, so ``http://host:notaport`` splits cleanly and only raises when something reads
    ``.port`` — which httpx does, per scenario, far from here.
    """
    try:
        parts = urlsplit(url)
        host, _port = parts.hostname, parts.port
    except ValueError as exc:
        return f'{type(exc).__name__}: {exc}'
    if parts.scheme.lower() not in ('http', 'https'):
        return 'expected an http(s):// url'
    if not host:
        return 'the url names no host'
    return None


def _resolve_observability_posture(args: argparse.Namespace) -> tuple[ObservabilityPosture, str]:
    """The run's observability posture and WHERE it was declared.

    Precedence, verbatim the ``.run_mode`` shape: an explicit flag (``--no-observability`` /
    ``--observability``) > ``$PLURAI_OBSERVABILITY`` > the ``<workdir>/.observability`` sidecar >
    the default, which is ``isolated``.

    The default matters. Reading sink selection straight off the environment would be unsafe:
    the environment is discovered by walking UP from the cwd to the nearest ``.env``, so a
    directory the customer owns could enable a production sink for a run nobody pointed at it,
    with the only trace being platform warnings mid-run. Recording is therefore something
    somebody SAYS, in one of three places; the sinks' own ``<SINK>__ENABLED`` flags decide
    WHICH sink, not whether.

    Every rung that can be SAID is answered where it is said. An unrecognised
    ``$PLURAI_OBSERVABILITY`` does not fall through to the sidecar: falling through would publish a
    ``posture_source`` naming a LOWER rung than the one the operator wrote in, so a typo would read
    on screen as a decision made somewhere the operator had not touched. A declaration this
    resolver cannot read is answered by the SAFE posture and says so.

    ``PLURAI_OBSERVABILITY`` is read from the real process environment ONLY, never from the
    discovered ``.env``, and that asymmetry with the sink flags is the point rather than an
    oversight: ``.env`` is found by walking UP from the cwd, so honouring a POSTURE from it would
    reinstate exactly the discovery this function exists to remove. A sink flag in that file may
    still choose WHICH sink, because by then somebody has already said whether.
    """
    if getattr(args, 'no_observability', False):
        return 'isolated', '--no-observability'
    if getattr(args, 'observability', False):
        return 'recording', '--observability'
    env_raw = (os.environ.get('PLURAI_OBSERVABILITY') or '').strip()
    if env_raw.lower() in get_args(ObservabilityPosture):
        return cast('ObservabilityPosture', env_raw.lower()), '$PLURAI_OBSERVABILITY'
    if env_raw:
        print(
            f'[observability] ignoring unusable $PLURAI_OBSERVABILITY ({env_raw[:60]!r}) — expected '
            f'{" or ".join(get_args(ObservabilityPosture))}. This run records NOTHING; the lower rungs are NOT '
            'consulted, because a declaration made here must not be answered by a file nobody named.',
            file=sys.stderr,
        )
        return 'isolated', '$PLURAI_OBSERVABILITY (unusable — defaulted to isolated)'
    raw = _sidecar_first_line(_agent_module_path(args), _OBSERVABILITY_SIDECAR)
    declared = raw.lower() if raw else ''
    if declared in get_args(ObservabilityPosture):
        return cast('ObservabilityPosture', declared), f'{_OBSERVABILITY_SIDECAR} in the agent workdir'
    if declared:
        print(
            f'[observability] ignoring unusable {_OBSERVABILITY_SIDECAR} ({declared[:60]!r}) — expected '
            f'{" or ".join(get_args(ObservabilityPosture))}. This run records NOTHING.',
            file=sys.stderr,
        )
        return 'isolated', f'{_OBSERVABILITY_SIDECAR} (unusable — defaulted to isolated)'
    return 'isolated', 'the default (recording is opt-in)'


def _declared_network_field(base: Settings, name: str) -> bool:
    """Did the env / ``.env`` explicitly set this ``NETWORK__*`` field?

    ``model_fields_set`` rather than a value comparison against the default: an operator who
    exports the default value on purpose has still DECLARED it, and a comparison would read that
    as unset and let a lower rung quietly replace it.

    Not an ``os.environ`` sniff: Pydantic resolves ``NETWORK__*`` from the real environment AND
    from the ``.env`` discovered by walking up from the cwd; ``os.environ.get`` sees only the
    first, so a url or a posture declared in ``.env`` would be invisible and the WORKDIR SIDECAR —
    the rung documented as lowest — would silently outrank it. One question, one answer, for all three fields.
    """
    return name in base.network.model_fields_set


def _gathered_exit_code(*, failures: bool, upload_ok: bool) -> int:
    """The exit code of a gathered run: a failed scenario outranks a dataset upload that did not complete."""
    if failures:
        return exit_codes.POLICIES_FAILED
    return exit_codes.OK if upload_ok else exit_codes.TRANSPORT


def _resolve_endpoint_or_exit() -> Endpoint:
    """The backend this run talks to (:func:`plurai._endpoints.resolve`), or its configuration error on stderr and the registry's exit code."""
    try:
        return resolve_endpoint()
    except EndpointError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(exit_codes.CREDENTIALS) from error


def _command_run_id_or_exit(flag: str | None) -> str | None:
    """The command run id (``--command-run-id``, else ``PLURAI_COMMAND_RUN_ID``), or a usage error naming the bad value's source."""
    try:
        return command_run.resolve(flag, os.environ)
    except ValueError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(exit_codes.USAGE) from error


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Resolve the global ``Settings`` carrier from the CLI args + process env.

    CLI flags override ONLY when explicitly passed (their argparse defaults are ``None``
    sentinels) — an absent flag falls through to the ``NETWORK__*`` env/.env value and
    then to the ``NetworkSettings`` default. The backend and its token are not settings: they
    come from :func:`plurai._endpoints.resolve` (``PLURAI_URL`` / ``PLURAI_TOKEN``, else the saved
    login), and a run without them stops with the missing-configuration message. Precedence per field:
    ``--http-timeout`` > ``NETWORK__HTTP_TIMEOUT`` > ``1200.0``;
    ``--agent-turn-timeout`` > ``NETWORK__AGENT_TURN_TIMEOUT`` > ``<workdir>/.turn_timeout``
    > the ``--run-phase`` default (1560s qualification / 2160s full) (the flag must NOT
    carry a numeric argparse DEFAULT: one would force the update on EVERY run and leave the
    documented env var permanently unreachable — hence the ``None`` sentinel; the PHASE is
    passed instead of a number for exactly that reason, so it can only ever lose to a value
    somebody declared);
    ``--scenario-timeout`` (alias ``--experiment-task-timeout``) >
    ``NETWORK__SCENARIO_TIMEOUT`` > the ``--run-phase`` default (1500s qualification /
    2400s full) — the same phase-lowest-rung shape as the per-turn budget above;
    ``--mock-mode`` > ``NETWORK__MOCK_MODE`` > ``<workdir>/.run_mode`` > ``'connect'``
    (the sidecar is the posture the setup flow DECLARED — lowest
    precedence, never sniffed from a failed connect);
    ``--mock-service-url`` > ``NETWORK__MOCK_SERVICE_URL`` > ``<workdir>/.mock_service_url``
    > ``None`` (nothing pinned it — the server's per-thread url is the only one). The SAME
    four-rung shape as the posture directly above it, and for the same reason: the truth
    already exists on disk, and requiring it again per invocation is the duplication that
    manufactures the failure;
    """
    endpoint = _resolve_endpoint_or_exit()
    register_backend_token(endpoint.token)
    base = Settings().with_endpoint(endpoint).with_command_run_id(_command_run_id_or_exit(getattr(args, 'command_run_id', None)))
    # The timeout/retry flags below are read via ``getattr`` so a sparse Namespace (e.g. a
    # unit-test stub that only sets mock_mode) falls back to
    # NetworkSettings' own defaults instead of raising; a fully-parsed args from
    # ``_parse_args`` always carries them.
    update: dict[str, object] = {}
    if args.mock_mode is not None:
        update['mock_mode'] = args.mock_mode
    elif not _declared_network_field(base, 'mock_mode'):
        sidecar = _sidecar_mock_mode(_agent_module_path(args))
        if sidecar is not None:
            update['mock_mode'] = sidecar
            print(f"[mock-mode] '{sidecar}' (declared in {_sidecar_display(args, _RUN_MODE_SIDECAR)}; pass --mock-mode to override)")
    # The deployed mock's URL, on the SAME four rungs as the posture above. Announced when the
    # sidecar supplies it, because a pinned url changes what an unreachable mock DOES (it
    # refuses the run — see _preflight_mock_service) and that must not be an invisible decision.
    mock_service_url = getattr(args, 'mock_service_url', None)
    if mock_service_url is not None:
        update['mock_service_url'] = mock_service_url
    elif not _declared_network_field(base, 'mock_service_url'):
        sidecar_url = _sidecar_mock_service_url(_agent_module_path(args))
        if sidecar_url is not None:
            update['mock_service_url'] = sidecar_url
            print(f'[mock-url] {sidecar_url} (declared in {_sidecar_display(args, _MOCK_SERVICE_URL_SIDECAR)}; pass --mock-service-url to override)')
    # Per-turn budget, in precedence order. The env var is NOT re-read here — pydantic already
    # applied it (and any ``.env``) to ``base``, so the question is whether that field was
    # DECLARED, which ``_declared_agent_turn_timeout`` answers from ``model_fields_set``.
    # Deliberately NOT a comparison against the field default: exporting the default value on
    # purpose is still a declaration, and a comparison would read it as unset and silently
    # lower it to the qualification default in a qualification pass. The sidecar and the phase default may only fill
    # a slot nobody else claimed.
    # Every rung names itself in ``turn_timeout_source``, announced once the value is validated —
    # the phase default is the rung most runs land on, so it must not be the one that is silent.
    agent_turn_timeout = getattr(args, 'agent_turn_timeout', None)
    if agent_turn_timeout is not None:
        update['agent_turn_timeout'] = agent_turn_timeout
        turn_timeout_source = '--agent-turn-timeout flag'
    elif _declared_network_field(base, 'agent_turn_timeout'):
        turn_timeout_source = 'NETWORK__AGENT_TURN_TIMEOUT (env / .env); pass --agent-turn-timeout to override'
    else:
        sidecar_timeout = _sidecar_turn_timeout(_agent_module_path(args))
        if sidecar_timeout is not None:
            update['agent_turn_timeout'] = sidecar_timeout
            turn_timeout_source = f'declared in {_sidecar_display(args, _TURN_TIMEOUT_SIDECAR)}; pass --agent-turn-timeout to override'
        else:
            phase: RunPhase = getattr(args, 'run_phase', 'full')
            update['agent_turn_timeout'] = default_agent_turn_timeout(phase)
            turn_timeout_source = f'--run-phase {phase} default; declare {_sidecar_display(args, _TURN_TIMEOUT_SIDECAR)} to change'
    # The per-scenario TIME budget, on the same rungs and for the same reason as the per-turn
    # budget above: flag > env (declaredness via ``model_fields_set``, never a value compare)
    # > the run-phase default. No sidecar rung — there is no ``.scenario_timeout`` sidecar, and
    # inventing one would add a declaration surface nobody asked for.
    scenario_timeout = getattr(args, 'scenario_timeout', None)
    if scenario_timeout is not None:
        update['scenario_timeout'] = scenario_timeout
    elif not _declared_network_field(base, 'scenario_timeout'):
        update['scenario_timeout'] = default_scenario_time_budget(getattr(args, 'run_phase', 'full'))
    # One flag for the ONE read timeout — it covers /threads, /user-turn AND /finalize
    # alike, and is not split into per-category timeouts.
    http_timeout = getattr(args, 'http_timeout', None)
    if http_timeout is not None:
        update['http_timeout'] = http_timeout
    retries = getattr(args, 'retries', None)
    if retries is not None:
        update['retries'] = retries
    # Rebuild rather than model_copy(update=...): model_copy SKIPS pydantic validation, so a
    # CLI value would bypass every field constraint the env path enforces (the seconds floor,
    # allow_inf_nan, retries >= 0). Rebuilding runs them; NetworkSettings is the ONE enforcer.
    try:
        net = NetworkSettings(**{**base.network.model_dump(), **update})
    except ValidationError as exc:
        first = exc.errors()[0]
        field = '.'.join(str(loc) for loc in first['loc'])
        print(f'ERROR: invalid value for network setting {field!r}: {first["msg"]} (got {first.get("input")!r})', file=sys.stderr)
        raise SystemExit(exit_codes.USAGE) from exc
    settings = base.model_copy(update={'network': net})
    print(f'[turn-timeout] {net.agent_turn_timeout:.0f}s ({turn_timeout_source})')
    _announce_scenario_budget(net)
    return _apply_observability_posture(settings, args)


def _announce_scenario_budget(net: NetworkSettings) -> None:
    """Say what bounds a scenario, and what expiry will DO to it.

    Always printed, because the budget is always on — and because its expiry is not
    fatal, which is the part an operator reading a run needs to know before they see a row come
    back ``incomplete``. The hard ceiling is stated beside it as the hang detector it is, so a
    ``harness_scenario_timeout`` in the artifacts is never mistaken for this budget firing.
    """
    budget = net.scenario_timeout or FULL_RUN_SCENARIO_TIME_BUDGET
    ceiling = derive_scenario_hard_ceiling(budget, net)
    # The variable below names no setting, and pydantic silently DROPS an unknown key (both models ignore
    # extras), so an operator who sets it would otherwise see the phase default and no explanation for it.
    if os.environ.get('NETWORK__SCENARIO_TIME_BUDGET'):
        print(
            '[scenario-budget] NOTE: NETWORK__SCENARIO_TIME_BUDGET is set but is not a setting — --scenario-timeout '
            'is the landing budget, and it is what the number below came from.',
            file=sys.stderr,
        )
    print(
        f'[scenario-budget] {budget:.0f}s per scenario (the dialog ENDS between turns and is still finalized, '
        f'so the row lands `incomplete` and is graded for the turns it reached; hard hang-ceiling {ceiling:.0f}s)'
    )


def _apply_observability_posture(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply the run's observability posture to ``settings``, and SAY what it did.

    Three outcomes, three different lines, because the remedies differ: isolation over a sink
    that IS enabled, recording to a sink, and recording declared with no sink to record to. The
    posture itself — and which rung declared it — is :func:`_resolve_observability_posture`.
    """
    posture, posture_source = _resolve_observability_posture(args)
    if posture == 'isolated':
        # Name what is being SUPPRESSED, not merely that isolation is on. A run whose sink was
        # enabled in the environment and is now off must say so on the line the operator reads,
        # or it reads as a platform that mysteriously stopped receiving runs.
        if settings.active is not None:
            print(
                f"[observability] ISOLATED: '{settings.active}' is enabled in the environment but this run records "
                f'NOTHING (posture from {posture_source}). Recording is opt-in — pass --observability, set '
                f'PLURAI_OBSERVABILITY=recording, or write "recording" into {_OBSERVABILITY_SIDECAR} beside simulate.py.'
            )
        return _with_observability_off(settings)
    if settings.active is not None:
        # Say WHERE the sink came from. Discovery walks up from the cwd, so a parent
        # directory's .env can turn recording on with nothing on the command line —
        # visible only as platform warnings mid-run. Read the real environment FIRST:
        # it outranks .env by source order, so naming the file whenever one merely
        # exists would point the operator at something that may not hold the flag.
        origin = 'the environment' if os.environ.get(f'{settings.active.upper()}__ENABLED') else (resolve_dotenv_path() or 'the environment')
        print(
            f"[observability] recording to '{settings.active}' (sink enabled via {origin}; posture declared by "
            f'{posture_source}; pass --no-observability to turn it off)'
        )
    else:
        print(f'[observability] posture "recording" ({posture_source}) but no sink is enabled — nothing will be recorded.')
    return settings


def _with_observability_off(settings: Settings) -> Settings:
    """Every sink disabled, whatever the environment asked for.

    Turns each sink config off rather than clearing ``active`` alone: the factory reads
    ``active``, but a sink config left enabled would still be the truth on the model that
    everything else inspects.
    """
    off = {name: getattr(settings, name).model_copy(update={'enabled': False}) for name in get_args(SinkName)}
    return settings.model_copy(update={**off, 'active': None})


#: A route behind the backend's bearer gate that names no real job: what it answers says only whether
#: this run's own credentials pass the gate (401) or not.
_CREDENTIAL_PROBE_PATH: Final = '/jobs/plurai-preflight-auth'


def _backend_auth_headers(settings: Settings) -> dict[str, str]:
    """The ``Authorization`` header :class:`OrchestratorClient` sends to this run's backend, or none without a token."""
    endpoint = settings.endpoint
    return endpoint.auth_headers(endpoint.url)


def _classify_healthz(response: httpx.Response) -> Literal['alive', 'outage', 'answer']:
    """What one ``/healthz`` answer says: ``alive`` (a 2xx, or a 402 naming a credit kind: the gateway refuses every route at an
    organization's hard stop, so something live is there and the run stops for credit on its first real call), ``outage`` (a
    502, 503 or 504: the backend is not there for now) or ``answer`` (any other status, which waiting does not change)."""
    if 200 <= response.status_code < 300:
        return 'alive'
    if response.status_code == CREDIT_REFUSED_STATUS and credit_block_from_body(response_json(response)) is not None:
        return 'alive'
    return 'outage' if response.status_code in TRANSIENT_STATUS_CODES else 'answer'


def _probe_healthz(settings: Settings, url: str) -> tuple[bool, httpx.Response | None, list[int]]:
    """Ask ``/healthz`` until it settles: ``(alive, last answer that was no outage, the statuses of those answers)``.

    A connection that never opens and 502, 503 or 504 answers are waited out under the transient budget of the request loop
    (about 60 seconds, so a backend upgrade passes), and a spent budget raises :class:`BackendUnavailable`, which ends the run as
    the unavailable stop. Any other non-2xx answer, and a host name that does not resolve (a mistyped URL, which waiting does
    not fix), is asked ``PREFLIGHT_ATTEMPTS`` times, a second apart, and a 3xx once; a name that never resolves leaves no answer.
    """
    retry = RetryConfig.from_settings(settings.network)
    attempts = max(retry.max_retries + 1, PREFLIGHT_ATTEMPTS)  # the probe count is the floor, whatever --retries says
    last_response: httpx.Response | None = None
    answered: list[int] = []
    outages = 0
    name_failures = 0
    while True:
        outage: UnavailableCause = 'transport'
        try:
            with _http.Client(timeout=httpx.Timeout(30.0, connect=PREFLIGHT_CONNECT_TIMEOUT), headers=_backend_auth_headers(settings)) as probe:
                response = probe.get(url)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if is_name_not_found(exc):
                name_failures += 1
                if name_failures >= PREFLIGHT_ATTEMPTS:
                    return False, last_response, answered
                time.sleep(PREFLIGHT_RETRY_DELAY)
                continue
        except Exception:  # noqa: BLE001 — connected-but-odd (read timeout, bad TLS record, …) proves the backend is THERE
            return True, last_response, answered
        else:
            verdict = _classify_healthz(response)
            if verdict == 'alive':
                return True, last_response, answered
            if verdict == 'answer':
                last_response = response
                answered.append(response.status_code)
                if 300 <= response.status_code < 400 or len(answered) >= PREFLIGHT_ATTEMPTS:
                    return False, last_response, answered  # a redirect is the same answer every time
                time.sleep(PREFLIGHT_RETRY_DELAY)
                continue
            outage = 'status'
        outages += 1
        if outages >= attempts:
            raise BackendUnavailable(UnavailableBlock(outage))
        time.sleep(backoff_seconds(outages, base=retry.backoff_base, cap=retry.backoff_cap))


def _preflight_backend(settings: Settings) -> GateFinding | None:
    """Fast-fail liveness probe against ``/healthz``, then one request with this run's own credentials.

    Down and slow are otherwise indistinguishable: with no preflight, a dead or wedged backend
    surfaces as per-request timeouts up to ``http_timeout`` per call, scenario after scenario.
    ``/healthz`` is trivially fast on a live backend, so probing answers "is the backend THERE" up
    front. The backend serves it unauthenticated, but a tenant proxy in front of it
    gates every route on the bearer before it looks the backend up, so the probe carries the run's own
    ``Authorization`` header (none without a token, which a tokenless loopback or bridge backend
    accepts). The answer is a gate finding or ``None``, bar the outage that ends the run:

    * the host name never resolved across ``PREFLIGHT_ATTEMPTS`` — ``preflight.backend_unreachable`` (invariant: nothing could run),
      naming the URL settings to check;
    * a connection that never opened, or every answer a 502, 503 or 504, across the transient retry budget (about 60
      seconds, the one the request loop waits out) — no finding: :class:`BackendUnavailable` is raised and the run ends as
      the unavailable stop (:func:`_probe_healthz`);
    * a 3xx — ``preflight.backend_redirected`` (invariant: the run client follows no redirect, so
      every call would get the same answer), naming the ``Location``;
    * a 401 or 403 on every attempt (``PREFLIGHT_ATTEMPTS``) — ``preflight.backend_auth_failed``, an advisory that HOLDS and
      names which status and whether a bearer went out: an auth layer refused the run, which says
      nothing about the backend's health;
    * any other non-2xx on every attempt — ``preflight.backend_unhealthy``, an advisory that HOLDS:
      a gateway in front of a working API answers this way too, so the plugin decides;
    * a 402 whose body names a credit kind is an answer from a live backend, so it counts as alive and the run stops for credit
      on its first real call;
    * anything that answered (a healthy backend, or either holding answer above, which it then outranks),
      then answering 401 to ONE request with this run's own headers on a bearer-gated route
      (:data:`_CREDENTIAL_PROBE_PATH`) — ``preflight.backend_rejects_credentials``
      (invariant: every call of the run would 401). This measures the bearer; it never guesses the
      deployment from the URL, so a tokenless internal bind and a wrong token are both told apart. The same
      request measures the client-version handshake: a 426 raises :class:`~plurai.client_handshake.UpgradeRequiredError`,
      which ``_main_body`` turns into the ``pip`` line and ``UPGRADE_REQUIRED``, because every call of the run would be refused.

    A connect that then stalls or breaks (loaded backend, cold TLS) reads as alive. One transient blip
    never decides: only every attempt failing does. Sync by design: it runs in ``main()``'s
    pre-spend slot, before the dispatch that chooses a strategy.
    """
    base = settings.endpoint.url.rstrip('/')
    health = '/healthz'
    url = base + health
    alive, last_response, answered = _probe_healthz(settings, url)
    if not alive and last_response is None:
        return make_finding(
            PREFLIGHT_BACKEND_UNREACHABLE,
            subject=base,
            evidence={'base_url': base, 'attempts': PREFLIGHT_ATTEMPTS},
            observed={'error': 'name not found'},
            remedy=f'backend unreachable: GET {url} could not resolve the host name across {PREFLIGHT_ATTEMPTS} attempts. The backend URL is '
            'wrong, or this machine cannot reach its DNS — nothing would run. Check the URL (PLURAI_URL, or `plurai login --status`), then retry.',
            draft_ref=None,
        )
    if not alive and last_response is not None:
        status = last_response.status_code
        # The probe carries the bearer, and an auth layer may echo it back in its error body.
        register_backend_token(settings.endpoint.token)
        body = redact_cli_credentials(describe_failure(last_response, limit=200))
        if 300 <= status < 400:
            location = last_response.headers.get('location') or ''
            return make_finding(
                PREFLIGHT_BACKEND_REDIRECTED,
                subject=base,
                evidence={'base_url': base, 'status': status, 'location': location},
                observed={'body': body},
                remedy=f'backend redirected: GET {url} answered {status} to {location or "<no Location>"}. The run client follows no '
                'redirect, so every call would get the same answer. Set PLURAI_URL to the URL it redirects to.',
                draft_ref=None,
            )
        cause = (
            'the backend, or a proxy in front of it, is failing or has no backend behind it'
            if status >= 500
            else 'something answered that is not a healthy Plurai backend'
        )
        # Something answered, so the credentials are measured too: an invariant 401 from the bearer-gated route
        # (a gateway in front of a working API that rejects this token) is the more decisive answer, and the
        # bearer goes only to this run's own base_url, where the run sends it anyway.
        if (rejected := _preflight_credentials(settings, base)) is not None:
            return rejected
        if all(code in (401, 403) for code in answered):
            return _backend_auth_failed(settings, base, health, status, body)
        return make_finding(
            PREFLIGHT_BACKEND_UNHEALTHY,
            subject=base,
            evidence={'base_url': base, 'path': health, 'status': status},
            observed={'body': body},
            remedy=f'backend unhealthy: GET {url} got no 2xx across {PREFLIGHT_ATTEMPTS} attempts; the last answer was {status}. /healthz answers '
            f'2xx on a live backend, so {cause}. Check the URL and the backend; if a gateway in front of a working API answers this route, '
            'overrule this finding by its fingerprint.',
            draft_ref=None,
        )
    return _preflight_credentials(settings, base)


def _backend_auth_failed(settings: Settings, base: str, path: str, status: int, body: str) -> GateFinding:
    """``/healthz`` answered 401 or 403 on every attempt: an auth refusal, named as one, never as an unhealthy box.

    The remedy says which of four cases it is (401 or 403, bearer sent or not), because each points at a
    different fix. The token itself goes nowhere but the request header.
    """
    token_sent = bool(settings.endpoint.token)
    url = base + path
    if status == 401 and token_sent:
        cause = f'401 Unauthorized to the bearer token this run sent: the auth layer in front of the backend does not accept it. Fix it: {CREDENTIAL_FIX}.'
    elif status == 401:
        cause = f'401 Unauthorized and this run sent no bearer token: the backend, or a proxy in front of it, requires one. Fix it: {CREDENTIAL_FIX}.'
    elif token_sent:
        cause = (
            '403 Forbidden to the bearer token this run sent: the token is known but not allowed this backend (another tenant '
            "or sandbox, or a revoked grant). Use a token issued for this backend's URL."
        )
    else:
        cause = f'403 Forbidden and this run sent no bearer token: the backend, or a proxy in front of it, refuses anonymous callers. Fix it: {CREDENTIAL_FIX}.'
    return make_finding(
        PREFLIGHT_BACKEND_AUTH_FAILED,
        subject=base,
        evidence={'base_url': base, 'path': path, 'status': status, 'token_sent': token_sent},
        observed={'body': body},
        remedy=f'backend auth failed, not unhealthy: GET {url} answered {cause} If only the health route is gated and the API '
        'works, overrule this finding by its fingerprint.',
        draft_ref=None,
    )


def _preflight_credentials(settings: Settings, base: str) -> GateFinding | None:
    """One request with this run's own headers to a bearer-gated route: a 401 means every call would 401, a 426 that the SDK is below the backend's floor.

    The bearer is sent only to ``base_url``, where the run sends it anyway. Anything other than a 401 —
    a 404 for the unknown job, a gateway's own answer, a transport error — says nothing about the
    credentials and proceeds.
    """
    try:
        with _http.Client(timeout=httpx.Timeout(30.0, connect=PREFLIGHT_CONNECT_TIMEOUT), headers=_backend_auth_headers(settings)) as probe:
            response = probe.get(base + _CREDENTIAL_PROBE_PATH)
    except Exception:  # noqa: BLE001 — no answer is no evidence about the credentials
        return None
    if response.status_code == client_handshake.UPGRADE_STATUS:
        # The route sits behind the version gate as well as the bearer gate, so a 426 here is the same refusal every call would get.
        raise client_handshake.UpgradeRequiredError(client_handshake.upgrade_message_for(response))
    if response.status_code != 401:
        return None
    token_sent = bool(settings.endpoint.token)
    return make_finding(
        PREFLIGHT_BACKEND_REJECTS_CREDENTIALS,
        subject=base,
        evidence={'base_url': base, 'token_sent': token_sent},
        observed={},
        remedy=(
            f'{base} rejects the bearer token this run sent (401 on a bearer-gated route), so every call would be refused. Fix it: {CREDENTIAL_FIX}.'
            if token_sent
            else f'{base} requires a bearer token and this run sent none (401 on a bearer-gated route). Fix it: {CREDENTIAL_FIX}.'
        ),
        draft_ref=None,
    )


def _mock_url_pin_source(args: argparse.Namespace, settings: Settings) -> str | None:
    """WHICH rung pinned this run's mock url, or ``None`` when nothing pinned one.

    Only the top two rungs need naming, and both are answerable without re-reading anything: the
    flag is on ``args``, and the env rung is answerable from a fresh ``Settings`` (pydantic
    records the field in ``model_fields_set`` whether the value came from the real environment or
    from the discovered ``.env`` — see :func:`_declared_network_field`). A url that is pinned and
    came from neither came from the workdir sidecar.

    The distinction exists because the two kinds of pin have opposite trust. A flag and an env
    var are acts of THIS invocation. ``.mock_service_url`` is a file five plugin sites write and
    none deletes — ``undeploy_env`` SIGKILLs the mock and leaves it behind — so a stale copy is
    the NORMAL state of a workspace between deploys, and the plugin's own documentation says as
    much. See :func:`_preflight_mock_service` for what each one does when the probe fails.
    """
    if settings.network.mock_service_url is None:
        return None
    if getattr(args, 'mock_service_url', None) is not None:
        return '--mock-service-url'
    if _declared_network_field(Settings(), 'mock_service_url'):
        return 'NETWORK__MOCK_SERVICE_URL'
    return _MOCK_SERVICE_URL_SIDECAR


def _preflight_mock_service(settings: Settings, *, pin_source: str | None = None) -> tuple[Settings, GateFinding | None]:
    """Check a PINNED mock url before the run spends anything. Returns ``(settings, invariant finding or None)``.

    Runs only when this workspace pinned a url (``--mock-service-url`` /
    ``NETWORK__MOCK_SERVICE_URL`` / ``.mock_service_url``) AND the posture is ``connect``. In
    ``local`` / ``mockless`` the child hosts its own loopback stub and a pinned url is a token
    rather than an address, so probing it would refuse a run that never needed it; with nothing
    pinned there is no claim to check — the server hands each thread its own url.

    WHAT AN UNREACHABLE PIN COSTS DEPENDS ON WHO PINNED IT, and that is the whole shape of this
    guard:

    * pinned by ``--mock-service-url`` or ``NETWORK__MOCK_SERVICE_URL`` — REFUSE. This invocation
      named that address; running it against something else is running a suite nobody asked for.
    * pinned by the ``.mock_service_url`` sidecar — WARN LOUDLY and DROP the pin, so the server
      supplies each thread its url. The file is written by five plugin sites and deleted by none,
      ``undeploy_env`` leaves it behind, and the plugin's own docs call it "a stale per-run handle
      — don't rely on it"; refusing on it would kill a run over a file nobody chose to trust.
      Nothing is lost by proceeding: the localhost fallback this guard prevents only applies
      when NOTHING pinned a url, and here the pin is dropped precisely because it answered nothing.

    The sidecar url is still READ, still USED and still ANNOUNCED — only "and it is unreachable"
    falls back instead of aborting.

    WHY IT SITS IN ``main()``'s fail-closed slot rather than beside the per-scenario check that
    already exists (``_gate_for_redrive``'s readiness probe): both run strategies do real,
    externally-visible work BEFORE the first scenario opens — the pre-flight dataset upload
    creates the provider dataset and its examples. That ordering is deliberate (LangSmith pins an
    experiment session's example set to the dataset version as of session creation, so the grid
    must exist before any session opens, and a 0/N run is exactly when it most needs to render).
    A run which CANNOT START must not get that far: an unreachable mock refuses here, so a doomed
    invocation uploads nothing into the customer's workspace.

    Only repeated CONNECT-level failures count as down (unlike :func:`_preflight_backend`,
    which also requires a 2xx). Any HTTP answer — 403, 404, 503 — proves something is listening,
    and those states have their own remedies further in (a redeploy, a restart that self-heals).
    The one answer that is decided here is a 401, whoever pinned the url: the probe already sends
    this run's own bearer, so a 401 means every tool call and world read would be refused
    (``preflight.mock_rejects_credentials``). Connected-but-odd states (a read timeout, a half-open socket, a broken protocol
    frame) prove it too. What is NOT waved through is a request that never became a network
    attempt at all — a malformed url, an unknown scheme, a broken proxy: those raise here and
    would otherwise raise identically inside every scenario, which is precisely the death this
    pre-flight promises to convert into one message.
    """
    url = settings.network.mock_service_url
    if url is None or settings.network.mock_mode != 'connect':
        return settings, None
    probe_url = url.rstrip('/') + '/health'
    headers = settings.endpoint.auth_headers(probe_url)
    last_error = 'ConnectError'
    for attempt in range(PREFLIGHT_ATTEMPTS):
        try:
            with _http.Client(timeout=httpx.Timeout(30.0, connect=PREFLIGHT_CONNECT_TIMEOUT), headers=headers) as probe:
                status = probe.get(probe_url).status_code
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = type(exc).__name__
            if attempt + 1 < PREFLIGHT_ATTEMPTS:
                time.sleep(PREFLIGHT_RETRY_DELAY)
            continue
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError):
            # Connected-but-odd. ConnectError/ConnectTimeout were taken above, so what reaches
            # here is a read/write timeout or a socket that opened and then misbehaved — the box
            # is THERE, and refusing on it would be this guard inventing failures.
            return settings, None
        except Exception as exc:  # noqa: BLE001 — everything else is an unusable url, not an unreachable one
            return settings, make_finding(
                PREFLIGHT_MOCK_URL_UNUSABLE,
                subject=url,
                evidence={'mock_url': url, 'pin_source': pin_source, 'exc_type': type(exc).__name__},
                observed={'error': str(exc)},
                remedy=_unusable_pin_message(probe_url, exc, pin_source),
                draft_ref=None,
            )
        if status == 401:
            token_sent = bool(headers)
            return settings, make_finding(
                PREFLIGHT_MOCK_REJECTS_CREDENTIALS,
                subject=url,
                evidence={'mock_url': url, 'token_sent': token_sent},
                observed={'pin_source': pin_source},
                remedy=f"mock service rejects this run's credentials: GET {probe_url} answered 401, so every tool call and world read would be "
                f'refused. Fix the credentials ({CREDENTIAL_FIX}), or re-deploy the env and refresh the url.',
                draft_ref=None,
            )
        return settings, None
    if pin_source == _MOCK_SERVICE_URL_SIDECAR:
        # WARN and DROP, never refuse: see this function's docstring. The pin goes with the
        # notice — leaving it set would keep dialing the address that just answered nothing.
        print(
            f'[mock-url] WARNING: nothing answered GET {probe_url} across {PREFLIGHT_ATTEMPTS} attempts '
            f'({last_error}), and that url came from the {_MOCK_SERVICE_URL_SIDECAR} sidecar in the agent workdir — '
            'a per-run handle the deploy flow writes and never deletes, so a stale one outlives the mock it named. '
            'DROPPING it: the server supplies each thread its own url. '
            f'If this run should have used a deployed mock, re-deploy the env (/use-env -> deploy_env) to refresh '
            f'{_MOCK_SERVICE_URL_SIDECAR}, or delete it.',
            file=sys.stderr,
        )
        return settings.model_copy(update={'network': settings.network.model_copy(update={'mock_service_url': None})}), None
    fallback = (
        f' Refusing rather than falling back to {DEFAULT_MOCK_SERVICE_URL}: that literal is what BOTH this package '
        'and the Plurai backend hand out when nobody configured a url, so the fallback would run the whole suite '
        'against a mock nobody deployed and report the agent for it.'
        if url.rstrip('/') != DEFAULT_MOCK_SERVICE_URL.rstrip('/')
        else ''
    )
    return settings, make_finding(
        PREFLIGHT_MOCK_UNREACHABLE,
        subject=url,
        evidence={'mock_url': url, 'pin_source': pin_source},
        observed={'error': last_error},
        remedy=f'mock service unreachable: GET {probe_url} refused every connection across {PREFLIGHT_ATTEMPTS} attempts '
        f'({last_error}), and this invocation PINNED that url via {pin_source or "its configuration"}. Nothing would '
        f'run against a real world.{fallback} Re-deploy the env (/use-env -> deploy_env) and refresh the url, or pass '
        '--mock-service-url / --mock-mode to say otherwise.',
        draft_ref=None,
    )


def _unusable_pin_message(probe_url: str, exc: Exception, pin_source: str | None) -> str:
    """The refusal for a pinned url that could not even be DIALED.

    Refused whichever rung pinned it, unlike an unreachable one: "unreachable" has a benign
    reading (the mock is down, the server can still supply a url), and "not a url" has none —
    every scenario would raise this same error deep inside httpx, one at a time, after the
    dataset upload. Nothing is being protected by letting that happen.
    """
    return (
        f'mock service url unusable: GET {probe_url} could not be attempted at all '
        f'({type(exc).__name__}: {exc}); it was pinned by {pin_source or "this run"}. Every scenario '
        'would die on this same error deep inside the HTTP client. Fix the url (--mock-service-url / '
        f'NETWORK__MOCK_SERVICE_URL / {_MOCK_SERVICE_URL_SIDECAR}) and retry.'
    )


#: The most trials one invocation may take — each is a complete run, all of them at once.
MAX_TRIALS = 10


def _trial_count(raw: str) -> int:
    """``--trials``: an int in ``1..MAX_TRIALS``, else a usage error."""
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_TRIALS:
        raise argparse.ArgumentTypeError(f'expected a whole number of trials from 1 to {MAX_TRIALS}, got {raw!r}')
    return value


def _trials_refusal(cfg: _RunConfig) -> str | None:
    """Why this invocation cannot take ``--trials N`` for N >= 2, with its one remedy; else ``None``."""
    if cfg.trials < 2:
        return None
    if cfg.run_phase == 'qualification':
        return '--trials takes independent samples of a FINAL run; qualification takes one sample per scenario — drop --trials.'
    if (mode := _replay_mode()) in ('replay', 'record'):
        return (
            f'PLURAI_REPLAY={mode} serves one recorded fixture per scenario and turn, so every trial would be the same sample '
            '— unset PLURAI_REPLAY or drop --trials.'
        )
    return None


def _replay_turn_only_refusal(cfg: _RunConfig) -> str | None:
    """Why this invocation cannot take ``--replay-turn-only``, with its one remedy; else ``None``.

    A single-hop replay runs ONLY the regression folders it names, and nothing may re-run one of them
    at a longer budget: the turn-cap re-run and an explicit stability retry both would.
    """
    if not cfg.replay_turn_only:
        return None
    if cfg.rerun_on_turn_cap:
        return '--replay-turn-only stops after the issue turn by design, so there is no turn cap to re-run — drop --rerun-on-turn-cap.'
    if cfg.stability_retry is True:
        return '--replay-turn-only takes its samples as --trials; a stability retry would add an unequal one — drop --stability-retry.'
    if not cfg.scenario_ids:
        return '--replay-turn-only replays recorded regression folders only — name them with --scenario-id.'
    if others := [sid for sid in cfg.scenario_ids if not is_regression_scenario(sid)]:
        return f'--replay-turn-only replays recorded regression folders only; {others} are not — run them without --replay-turn-only.'
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = exit_codes.UsageParser(prog='plurai-run-simulation', description='Client-side end-to-end simulation runner.')
    parser.add_argument('--scenario-set-id', required=True, help='UUID of the scenario set in backend storage.')
    parser.add_argument(
        '--scenario-id',
        action='append',
        default=None,
        metavar='SCENARIO_ID',
        help='Scenario id within the set to run. Repeat for multiple scenarios. Omit entirely to run EVERY scenario in the set (enumerated from the server).',
    )
    parser.add_argument('--agent-module-path', required=True, type=expand_user_path, help="Path to the agent's simulate.py.")
    parser.add_argument('--out', required=True, type=expand_user_path, help='Output directory for per-scenario artifacts.')
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Discard a run already measured into --out. Without it, running the WHOLE set into a directory that '
        'already holds a run_summary.json is refused rather than silently overwriting that run. A targeted '
        '--scenario-id re-run never needs this: it merges into the existing summary.',
    )
    parser.add_argument(
        '--no-run-pointer',
        action='store_true',
        help="Leave the workspace's .last_run pointer where it is. For a side run into its own --out (e.g. the "
        'regression replays) that later commands must not read as the latest run.',
    )
    parser.add_argument(
        '--mock-service-url',
        default=None,
        help='Deployed mock-service URL. When omitted: $NETWORK__MOCK_SERVICE_URL, else the url the setup/deploy '
        'flow persisted in `.mock_service_url` beside simulate.py, else the server-supplied per-thread url. A url '
        'pinned by ANY of those three is health-probed before the run and REFUSES when nothing is serving it.',
    )
    parser.add_argument(
        '--mock-mode',
        choices=('connect', 'local', 'mockless'),
        default=None,
        help="Mock posture: 'connect' (default) talks to a deployed mock and FAILS if it's "
        "unreachable; 'local' hosts an in-process loopback mock with your @plurai.mock handlers "
        "(for local dev / tests); 'mockless' declares that NOTHING is mocked — a tool-less agent, "
        'or one whose every tool runs mode: real, so no surface is held constant. When omitted: '
        '$NETWORK__MOCK_MODE, else the posture the setup '
        "flow persisted in `.run_mode` beside simulate.py, else 'connect'. Never sniffed from a "
        'failed connect.',
    )
    parser.add_argument(
        '--command-run-id',
        default=None,
        help=(
            'Id that groups the backend jobs of one command, sent with the requests that start work. '
            f'Falls back to {command_run.ENV_VAR}; a run with neither is not grouped.'
        ),
    )
    parser.add_argument('--eval-set-id', default=None, help='UUID of an eval set in backend storage (optional).')
    parser.add_argument(
        '--max-turns',
        type=int,
        default=None,
        help='Hard cap on dialog turns, applied to every scenario. Omitted, it is derived from the '
        "scenarios this run plays: the longest one's published minimum turns + 2 + max(2, its complexity "
        'level), +1 when the agent opens the conversation, and never below 12 (12 when the floors cannot '
        'be read). One cap for the whole run, sized for its longest scenario: a scenario cut off mid-arc '
        'scores the obligations it never reached. The value and its source are recorded in run_summary.json.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Max scenarios to run concurrently (bounded asyncio.gather). Each scenario '
        'spawns an agent subprocess, so this caps in-flight subprocesses + LLM load. '
        'Default 4; pass 1 for fully serial execution. A bench declaring world_state.owner: external (or owner: agent with world_state.slots) '
        'runs at most world_state.slots at once (1 when undeclared or 1).',
    )
    parser.add_argument(
        '--trials',
        type=_trial_count,
        default=1,
        metavar='N',
        help='Run the whole set N times (1-10), CONCURRENTLY (in turn for world_state.owner: external, a declared world_state.slots, or when '
        'N x C agent processes do not fit the sandbox memory limit), as N independent complete runs: trial 1 is --out, '
        'trial k is --out/trial_k/. run_summary.json then carries a `trials` block: per scenario and per policy how many '
        'trials passed / failed / did not apply / were deferred to an owner policy / were inconclusive, labelled consistent_pass, '
        'consistent_fail, inconsistent, insufficient, not_applicable or deferred, plus the run-level count of scenarios that passed on every valid trial (at '
        'least two valid) out of those whose goal was measured, the per-trial pass rate with a Wilson 95%% interval, and how '
        'often the goal itself was reached. '
        '--concurrency stays per run, so N x C agent processes run at once. Default 1 (one run). Refused with '
        '--run-phase qualification and under PLURAI_REPLAY=replay|record; with an experiment-runner sink (Arize) the run takes 1 trial.',
    )
    parser.add_argument(
        '--agent-turn-timeout',
        type=float,
        default=None,
        help='Per-agent-turn wall-clock budget in seconds before the child agent is '
        'terminated (AgentTurnTimeout). Raise it for agents that make many '
        'LLM calls within a single turn — e.g. bulk generation tools. Defaults to the '
        '--run-phase budget (1560 qualification / 2160 full). Highest precedence: beats '
        'NETWORK__AGENT_TURN_TIMEOUT and the workdir .turn_timeout sidecar.',
    )
    parser.add_argument(
        '--run-phase',
        choices=('qualification', 'full'),
        default='full',
        help="Which phase this invocation is. 'qualification' is the fail-fast gate over a "
        'small sample: a 1560s per-turn budget AND no re-drive on a stalled turn, so the wall '
        "is really 26 minutes rather than 2 x 26. 'full' (default) is the scored run: a 2160s "
        'budget, re-drive intact. This selects DEFAULTS ONLY — it never overrides a budget '
        'declared by flag, env or sidecar.',
    )
    parser.add_argument(
        '--replay-turn-only',
        action='store_true',
        help='SINGLE-HOP regression replay: each named regression_ folder runs only its issue turn — the recorded '
        'conversation is seeded, the cut user message is replayed, the agent answers it (tool calls included) and the '
        'run stops. No simulated user, no turn cap; --max-turns is ignored. End-of-run checks are graded on that one '
        'turn and told it was cut. Requires --scenario-id regression_* ids; refused with --rerun-on-turn-cap and '
        '--stability-retry.',
    )
    parser.add_argument(
        '--replay-plan',
        type=expand_user_path,
        default=None,
        metavar='PLAN_JSON',
        help='How each recorded regression in this run is played: a JSON file {"single_hop": [ids], "full": [ids]}. A '
        'single_hop id runs only its issue turn, as --replay-turn-only would, beside every other scenario played whole; '
        'its stability retry and turn-cap re-run are skipped and --trials are its samples. A regression_ id in neither '
        'list is played whole with a warning; an unreadable plan is warned about and every regression is played whole.',
    )
    parser.add_argument(
        '--rerun-on-turn-cap',
        action='store_true',
        help='Re-run a scenario ONCE at double --max-turns when its conversation ran out of turns with the '
        "user simulator's own target verdict still unmet. Such a scenario scores low for obligations its "
        'dialog never reached, so its result is a statement about the budget rather than about the agent. '
        'Off by default: a re-run costs a full scenario, which is an operator decision. Both attempts are '
        'recorded — the superseded one under previous_attempt/ in the scenario folder.',
    )
    parser.add_argument(
        '--stability-retry',
        dest='stability_retry',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Take ONE second sample of a FAILED scenario to establish whether its failure reproduces. '
        'The first attempt STAYS AUTHORITATIVE — retrying only failures is statistically biased, so a passing retry '
        'never rescues a failing run; it only annotates the row `stable` / `flaky` / `unmeasured` in run_summary.json. '
        'Defaults ON in both phases; WHICH sample is taken is --stability-sample. A run INVALIDATED by a repeated '
        'simulated-user turn is re-run regardless of this flag: it measured nothing, so its retry is the first sample '
        'rather than a second.',
    )
    parser.add_argument(
        '--stability-sample',
        dest='stability_sample',
        choices=('judge_replay', 'agent_rerun'),
        default=None,
        help='WHICH second sample --stability-retry takes. `agent_rerun` re-runs the scenario end to end — a '
        'fresh agent output — and answers whether the failure is a coin flip of the agent; it is the only kind that can '
        "read `stable`, the verdict that licenses a redraw. `judge_replay` re-judges the FIRST attempt's own transcript "
        'with the same eval set (judge tokens only, no agent re-run) and answers whether each red is judge noise; a '
        'judge that grades the same input both ways has that cell named on the stability row. Default: `agent_rerun` for --run-phase '
        'qualification (at most every scenario), `judge_replay` for a full run (at most every scenario); an explicit '
        '`agent_rerun` on a full run retries at most half the suite.',
    )
    parser.add_argument(
        '--shape',
        choices=(DIALOG_SHAPE, PIPELINE_SHAPE),
        default=None,
        help='OVERRIDE which route drives this bench, for debugging. `dialog` is the /user-turn loop; `pipeline` sends '
        "ONE invoke_pipeline(inputs) call built from the scenario's injected agent context and scores the (inputs, "
        'outputs) pair, with no user simulator and no turns. Normally you should NOT pass this: the bench DECLARES its '
        "shape in its own plurai.config.yaml (`shape: dialog|pipeline`), which travels with the harness, and simulate.py's "
        'exported hooks are a HINT, not a gap-filler: a bench that exports invoke_pipeline and NO invoke_agent and declares no '
        '`shape:` is REFUSED at run start; one that exports BOTH is warned and dispatched as `dialog`. Passing it here overrides '
        'an UNDECLARED bench; one that contradicts a valid declaration WINS and is recorded as a finding.',
    )
    parser.add_argument(
        '--egress-probe',
        choices=('connect', 'dns', 'off'),
        default='connect',
        help='How hard to test that the agent under test CANNOT reach the production hosts declared in '
        "plurai.config.yaml's `egress` block. `connect` (default) opens one TCP handshake per declared "
        'host and closes it immediately — no bytes are written, no TLS is negotiated and no credential is ever '
        'loaded or sent. `dns` resolves the name only and sends nothing to the host itself. `off` touches the '
        'network not at all and just reports the declaration; neither of those two can show a host reachable, so the '
        'run records an egress.probe_mode_reduced finding. A reachable declared host that nothing enforces isolation '
        'for HOLDS the run until an --overrule names its finding; every other verdict ships with its finding.',
    )
    parser.add_argument(
        '--scan-contract',
        default=None,
        type=expand_user_path,
        help='A scan_contract.json written by `plurai-scan-contract check`. At run start its recorded source inventory is '
        'compared against the CURRENT --scan-corpus files, so a suite built over documents that have since changed is '
        'named as such instead of scoring the agent against a stale premise. Absent ⇒ the check reports '
        'NOT CHECKED and the run proceeds.',
    )
    parser.add_argument(
        '--scan-corpus',
        action='append',
        default=None,
        type=expand_user_path,
        help='Corpus root to re-hash for the --scan-contract drift check. Repeatable, in the same order as the recording `plurai-scan-contract check`.',
    )
    parser.add_argument(
        '--scan-suffixes',
        action='append',
        default=None,
        help='Extra file suffix the recording `plurai-scan-contract check` admitted (`--suffixes html` there ⇒ '
        '`--scan-suffixes html` here). Repeatable, additive to the default set. Needed only for a baseline that recorded '
        'no scope: a baseline that recorded one is re-walked at it and this is ignored. For a scope-less baseline the '
        're-check must walk the SAME scope it was taken at — a mismatch is detected and the comparison ABSTAINS with the '
        'reason rather than reporting drift, but an abstention is not a check, so pass the set it was recorded with.',
    )
    parser.add_argument(
        '--scenario-timeout',
        '--experiment-task-timeout',
        dest='scenario_timeout',
        type=float,
        default=None,
        help="Per-scenario TIME BUDGET in seconds, across all a scenario's turns. Expiry LANDS the "
        'scenario rather than killing it: the dialog ends between turns, the closing turn is still '
        'delivered, /finalize runs, the judges grade the turns that were reached, and the row is '
        'recorded `incomplete` (neither passed nor failed) with reason `time_budget_exhausted`. '
        'Defaults to 1500s (25 min) with --run-phase qualification and 2400s (40 min) with full. '
        'Applies on BOTH run strategies, so "may a scenario run forever" does not depend on which '
        'observability backend is configured. A separate HARD ceiling, derived from this and not settable, still cancels a '
        'genuinely hung attempt. --experiment-task-timeout is an alias for this flag.',
    )
    parser.add_argument(
        '--http-timeout',
        type=float,
        default=None,
        help='HTTP READ timeout in seconds, applied to every orchestrator call alike — /threads, '
        '/user-turn AND the finalize submit/poll (connect is separate, clamped at 10s, so a dead '
        'host still fails in seconds). Default 1200 (NetworkSettings). The call that can '
        'legitimately sit near that ceiling is POST /user-turn — one agent turn plus the '
        'server-side user-sim LLM. Finalize does NOT need it: it is a submit + a poll loop, so '
        'every individual request is fast and the multi-minute eval wait is bounded by '
        'NETWORK__FINALIZE_POLL_MAX_WALLCLOCK_SECONDS (default 1800) instead — that, not this, is '
        'the knob for a slow eval set. Raise THIS, not --retries, when POST /user-turn dies with a '
        'ReadTimeout: that call is deliberately not retried (the turn may already have been applied '
        'server-side). Note --scenario-timeout bounds a WHOLE scenario on both run strategies and '
        'defaults to 1500s with --run-phase qualification and 2400s with full, so it can bind '
        'before this one — raise it too when a single scenario legitimately runs long. Overrides '
        'NETWORK__HTTP_TIMEOUT.',
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=None,
        help='Transient-HTTP retry budget per request (jittered exponential backoff). Default 6 — '
        'waits of 1, 2, 4, 8, 16, 30s, i.e. a ~61-66s window that outlasts a backend upgrade; raise it on a flaky network, '
        'or 0 for one attempt, with which a single 502/503/504 or refused connection mid-run is already an outage '
        '(the scenario is re-driven once the backend answers again, and the run stops only if it does not). '
        'It covers the requests of a run (the mid-run calls); the liveness preflight probe at run start keeps its own '
        'minimum number of attempts whatever this says. What it covers is '
        'RISK-SCOPED, not every failure: connection errors always; 502/504 and timeouts only on '
        'requests marked idempotent — the GETs plus the dedup-backed finalize submit/poll/cancel; '
        '503 also on POST /threads and /user-turn, where only the front proxy emits one (provably '
        'never forwarded), but not on POST /fail. A read timeout on POST /user-turn is deliberately '
        'NOT retried — the turn may already have been applied server-side and there is no turn-level '
        'dedupe, so a retry would duplicate it; raise --http-timeout for those, not this. '
        'Exact policy: plurai/orchestrator/_retry.py. Overrides NETWORK__RETRIES.',
    )
    parser.add_argument(
        '--upload-datasets',
        action='store_true',
        help='BEFORE running, fetch the scenario set + upload it as a dataset to the '
        'active CLIENT-side observability sink (gated on your sink keys; no keys ⇒ '
        'no-op). Best-effort: a failed or partial upload does not abort the scenarios '
        'but is reflected in the exit code (nonzero on upload failure). Provider handles '
        f'are persisted to <out>/{DATASET_UPLOAD_METADATA_FILENAME} for idempotent re-runs. '
        'Implied for sinks that must have a populated dataset before the run (LangSmith); '
        'there the upload still happens but never affects the exit code.',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Explicit experiment/dataset-run name (both strategies). Defaults to a unique '
        'per-invocation stamp (also settable via $PLURAI_EXPERIMENT_RUN_NAME).',
    )
    parser.add_argument(
        '--agent-version',
        default=None,
        help="Your agent's build/version tag (e.g. a git SHA or semver). Sent on the "
        'create-thread wire AND stamped onto every observability record (trace metadata) so '
        'runs are comparable across agent versions. When omitted, defaults to '
        '`git describe --tags --always --dirty` (or the literal "unknown" outside a git repo).',
    )
    add_overrule_argument(parser)
    observability = parser.add_mutually_exclusive_group()
    observability.add_argument(
        '--no-observability',
        action='store_true',
        help='Record nothing to any observability platform for this run, whatever the '
        'environment says. This is also the DEFAULT posture — the flag remains for making it '
        'explicit (and for overriding a workspace that declared `recording`).',
    )
    observability.add_argument(
        '--observability',
        action='store_true',
        help='Record this run to the enabled <SINK>__ENABLED sink. Required because recording is '
        'OPT-IN: sink flags are commonly set in a .env discovered by walking UP from the cwd, so a '
        "run could be recorded into a customer's production workspace because of a parent directory "
        'nobody was thinking about. Also settable as PLURAI_OBSERVABILITY=recording, or once per workspace '
        'by writing `recording` into `.observability` beside simulate.py.',
    )
    return parser.parse_args(argv)


def _line_buffer_stdout() -> None:
    """Switch stdout to LINE buffering, best-effort, before anything prints.

    Python block-buffers stdout when it is a pipe, which is how the runner is always invoked
    (the plugin tees it to a log): the whole run's progress then arrives in one block AFTER the
    process exits, so a healthy multi-scenario run and a wedged one produce byte-identical
    output for minutes at a time, and a startup line can land after the child's stderr.

    Wrapped, and ``getattr`` rather than a direct call: ``reconfigure`` is a ``TextIOWrapper``
    method, not part of the ``TextIO`` protocol, so a captured or wrapped stdout (a test's
    StringIO, an exotic wrapper) may simply not have it — and a stream that cannot be
    reconfigured must not be a reason a run refuses to start.
    """
    reconfigure = getattr(sys.stdout, 'reconfigure', None)
    try:
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    except (AttributeError, ValueError, OSError):
        logger.debug('stdout could not be switched to line buffering; progress output may be block-buffered.', exc_info=True)


def main(argv: list[str] | None = None) -> int:
    """Strategy selector: a sink implementing the OPTIONAL ``ExperimentRunnerSink`` role
    runs the set as ONE provider experiment; every other (or no) sink takes the native
    bounded-gather path.

    Both strategies share the typed ``cfg`` + resolved ``settings`` + the single active
    ``sink`` built here. The event-loop boundary is visible AT THE DISPATCH: the native
    path runs inside ONE ``asyncio.run``, while the experiment path must stay sync at
    top level and owns its own loop boundaries (see :func:`_run_as_experiment`).
    """
    utf8_stdio()
    _line_buffer_stdout()
    try:
        with redacted_process_output():
            return exit_codes.guarded(RUNNER_COMMAND, lambda: _main(argv))
    except KeyboardInterrupt:
        print('Interrupted: the run was stopped.', file=sys.stderr)
        return exit_codes.INTERRUPTED


def _main(argv: list[str] | None) -> int:
    args = _parse_args(argv)
    # Read before anything else runs: a malformed overrule file refuses the run rather than reading as
    # "nothing was overruled", which is a different instruction from the one the caller wrote.
    try:
        overrules_file = load_overrules(args.overrule) if args.overrule is not None else None
    except OverruleFileError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return exit_codes.REFUSED
    return run_invocation(command=RUNNER_COMMAND, overrules_file=overrules_file, record=None, body=lambda: _main_body(args))


def _main_body(args: argparse.Namespace) -> int:
    """:func:`main` once its gate-authority invocation is open (see :mod:`plurai.gates.invocation`).

    The gate record is CLAIMED below, once this run owns ``--out``: every rung above that point
    leaves the directory exactly as it found it, so none of them writes a record there.
    """
    settings = _settings_from_args(args)
    gates = current_gate_authority()
    # Liveness, then this run's own credentials against a bearer-gated route, in ONE slot covering both
    # dispatch strategies below (the experiment path does even more up-front work — scope resolution +
    # a full dataset upload — before a dead backend would surface). A measurement, never a guess from
    # the URL: an invariant answer stops here; an unhealthy /healthz is a holding finding the funnel
    # below decides once every rung has reported.
    outage: BackendUnavailable | None = None
    try:
        backend = _preflight_backend(settings)
    except client_handshake.UpgradeRequiredError as refusal:
        print(f'ERROR: {refusal}', file=sys.stderr)
        return exit_codes.UPGRADE_REQUIRED
    except BackendUnavailable as refused:
        # The backend never answered within the transient budget. The run still claims --out like every other run-start stop
        # (below), so the stop is recorded where the plugin reads it; no further preflight asks the dead backend anything.
        outage, backend = refused, None
    if backend is not None:
        gates.emit(backend)
        if (code := _invariant_exit(None)) is not None:
            return code
    # The MOCK's liveness, in the same slot and for a sharper reason: below this line both
    # strategies upload the provider dataset before a single scenario opens, so a run that
    # cannot start would otherwise leave a dataset behind in the customer's workspace. See
    # _preflight_mock_service — it fires only for a workspace that PINNED a url, and what it
    # DOES about an unreachable one depends on which rung pinned it (a stale sidecar is dropped
    # with a warning, an explicit flag or env var is refused). It may hand back settings with the
    # pin removed, so the rest of main() reads what it returned, not what went in.
    mock_finding = None
    if outage is None:
        settings, mock_finding = _preflight_mock_service(settings, pin_source=_mock_url_pin_source(args, settings))
    if mock_finding is not None:
        gates.emit(mock_finding)
        if (code := _invariant_exit(None)) is not None:
            return code
    cfg = _RunConfig.from_args(args, mock_service_url=settings.network.mock_service_url)
    # Before --out is touched: the --trials postures that cannot take N samples (qualification, replay).
    if (trials_message := _trials_refusal(cfg)) is not None:
        print(f'ERROR: {trials_message}', file=sys.stderr)
        return exit_codes.USAGE
    if (replay_message := _replay_turn_only_refusal(cfg)) is not None:
        print(f'ERROR: {replay_message}', file=sys.stderr)
        return exit_codes.USAGE
    # A sink that drives the whole set as ONE vendor experiment records one sample per scenario, so
    # the run takes one trial — resolved HERE, before the rungs that read --out judge it at N.
    sink: ObservabilitySink | None = None
    if cfg.trials > 1 and isinstance(sink := create_observability_sink(settings), ExperimentRunnerSink):
        print('[trials] the Arize experiment sink records one sample per scenario — this run takes 1 trial')
        cfg = dataclasses_replace(cfg, trials=1)
    # THE DIRECTORY, and FIRST of the rungs that read --out — it is the only one whose failure
    # mode is destroying an artifact rather than merely running a set that should not have run,
    # so it precedes the three below that just refuse. One call site covers BOTH dispatch
    # strategies, and it must land above them: each one's entry CLEARS a full run's prior
    # summary (_clear_stale_run_summary), so by the time either strategy could notice, the
    # measurement this rung protects is already gone. Deliberately NOT _publish_early_exit:
    # every sibling below re-stamps the summary, and the whole point here is that --out is
    # left byte-identical.
    if (occupied_message := _occupied_out_refusal(cfg)) is not None:
        # Invariant, and reported on stderr only: no record is claimed yet, so --out stays byte-identical.
        gates.emit(
            make_finding(
                RUN_OUT_OCCUPIED,
                subject='out',
                evidence={'partial': cfg.is_partial, 'replace': cfg.replace},
                observed={'out': str(cfg.out)},
                remedy=occupied_message,
                draft_ref=None,
            )
        )
        return _invariant_exit(None) or exit_codes.REFUSED
    if (trials_out_message := _trials_out_refusal(cfg)) is not None:
        print(f'ERROR: {trials_out_message}', file=sys.stderr)
        return exit_codes.USAGE
    # THE GATE RECORD. From here on this invocation owns --out, so its record is written there on every
    # exit path, an early exit and an exception included; the previous invocation's record goes now.
    # It sits BESIDE run_summary.json, never inside it: that file's presence is the occupied-out
    # predicate above, ingest-run uploads it verbatim, and a --scenario-id run merges it — while a
    # gate record belongs to one invocation, carries overrule reasons, and never leaves the machine.
    current_gate_authority().claim(RecordTarget(path=cfg.out / GATE_AUTHORITY_FILENAME, when_empty=True, prepare=None))
    if outage is not None:
        return _stop_for_outage_before_any_scenario(cfg, outage)
    # THE THREE DECLARATIONS BESIDE THE HARNESS, read in ONE guarded block and dispatched on
    # below. The `try` sets the posture for the config that cannot testify AT ALL:
    # `discover_config_documents` raises on a config that exists and is not a
    # regular file, cannot be decoded, is unparseable YAML, has a non-mapping top level, nests a
    # declaration under a non-mapping, spells an illegal value or declares one twice and
    # disagrees. The console script is bare, so an uncaught raise would give the
    # operator a raw traceback and exit 1, the code THIS file reserves for "ran, some
    # rows failed" (see _write_run_summary), for a run that never started. A driver branching on
    # the code would then proceed to the report leg against an --out with nothing fresh in it.
    #
    # The same shape `_default_agent_version` uses, on the same file:
    # it catches the pair and degrades loudly rather than letting `plurai_sim/` at mode 000 reach
    # the operator as a stack trace. `workspace_config` says a raw traceback is "the one
    # presentation this module exists to rule out"; this rung is where that is honoured for the
    # declarations, and it is what makes the ordering comment below true.
    #
    # The reads are EAGER rather than short-circuited per rung:
    # all four answer off the same two local files with no I/O of their own beyond them, and a
    # config that stops one from answering stops the run whichever rung met it first.
    try:
        shape_message = undeclared_pipeline_refusal(cfg.agent_module_path, override=cfg.shape)
        contradicted = contradicted_shape_declaration(cfg.agent_module_path, override=cfg.shape)
        world_finding = unreset_external_world_finding(cfg.agent_module_path)
        # The explicit opt-in to a shared world (`world_state.shared_world_ok: true`): it lifts the hold
        # above and the run carries this named risk instead.
        shared_world = shared_world_risk(cfg.agent_module_path)
        unobservable_message = unobservable_world_warning(cfg.agent_module_path)
        world_owner = _declared_world_owner(cfg.agent_module_path)
        # How many copies of an external or agent-owned world the bench runs (``world_state.slots``):
        # refused here, exit 2 and before any spend, beside a mock/undeclared owner, as a non-count, on
        # disagreement, or when no .py beside the harness names PLURAI_WORLD_SLOT (every copy would be copy 0).
        world_slots = _declared_world_slots(cfg.agent_module_path)
        # What only ONE agent process can hold (``harness.single_instance``, e.g. a fixed bound port): refused
        # here, exit 2, when it is not a list of resource names.
        single_instance = _declared_single_instance(cfg.agent_module_path)
        _, surface_defects = discover_surface_types(cfg.agent_module_path)
        # …and what the run could read of the world it scores, recorded for run_summary.json off
        # the same reads while they are in hand (:func:`_record_world_observability`).
        _record_world_observability(cfg.agent_module_path, reason=unobservable_message, slots=world_slots, shared_world=shared_world)
    except UserSurfacesConfigError as exc:
        # The user-surfaces block cannot be parsed as a whole: no capture could be scoped.
        declaration = 'world_state.user_surfaces'
        evidence = {'declaration': declaration, 'exc_type': type(exc).__name__}
        gates.emit(make_finding(WORLD_USER_SURFACES_MALFORMED, evidence=evidence, **_unreadable_declaration(declaration, exc)))
        return _invariant_exit(cfg) or exit_codes.REFUSED
    except (WorkspaceConfigError, OSError) as exc:
        evidence = {'config': CONFIG_FILENAME, 'exc_type': type(exc).__name__}
        gates.emit(make_finding(CONFIG_UNREADABLE, evidence=evidence, **_unreadable_declaration(CONFIG_FILENAME, exc)))
        return _invariant_exit(cfg) or exit_codes.REFUSED
    # THE SHAPE: an invoke_pipeline-only bench that declares no `shape:` has no hook for the turn loop to
    # drive, so every scenario of it dies on turn 1 in the child. Invariant: the remedy is the declaration.
    if shape_message is not None:
        gates.emit(
            make_finding(
                HARNESS_PIPELINE_UNDECLARED,
                subject=cfg.agent_module_path.name,
                evidence={'harness': cfg.agent_module_path.name, 'hooks': ['invoke_pipeline']},
                observed={},
                remedy=shape_message,
                draft_ref=None,
            )
        )
        return _invariant_exit(cfg) or exit_codes.REFUSED
    # THE SHAPE AGAIN, from the other side: `--shape` contradicts a valid declaration. The flag is the
    # operator's explicit decision and the resolver already lets it win, so the run proceeds with it and
    # the contradiction is a finding that holds nothing.
    if contradicted is not None:
        word, source = contradicted
        gates.emit(
            make_finding(
                HARNESS_SHAPE_OVERRIDE_CONTRADICTS_DECLARATION,
                subject='shape',
                evidence={'declared': word, 'override': cfg.shape, 'config': _relative_to_bench(source, cfg)},
                observed={},
                remedy=f'--shape {cfg.shape} overrides `shape: {word}` declared in {source}; the run is dispatched as {cfg.shape}. Drop the '
                'flag to run the bench as it declares itself, or change the declaration if it is the wrong one.',
                draft_ref=None,
            )
        )
    # THE WORLD: a world the mock does not own and no reset clears. Its only other symptom is a GREEN run
    # scored on a predecessor's writes, so it HOLDS at the funnel until an overrule says what was verified.
    if world_finding is not None:
        gates.emit(world_finding)
    # ...unless the operator opted into that shared world: the run proceeds, and says what it carries.
    if shared_world is not None:
        print(
            f'WARNING: named risk {SHARED_WORLD_RISK_KIND}: {shared_world["display_name"]} (world_state.shared_world_ok: true; '
            f'live writers: {", ".join(shared_world["live_writers"]) or "none declared"}, owner: {shared_world["owner"]}). '
            'The suite caps at qualified and stability retries / trial agreement are not comparable.',
            file=sys.stderr,
        )
    # ...and the SECOND thing that world declaration implies, which is NOT a gate: a world nothing in this
    # run can READ. The dialog and every llm_as_judge policy still score; world-reading ones are demoted to
    # not-measured. It prints and FALLS THROUGH.
    if unobservable_message is not None:
        print(f'WARNING: {unobservable_message}', file=sys.stderr)
    # One user-surface entry whose paths name no entity type: that entry is not captured, the others are.
    for defect in surface_defects:
        gates.emit(
            make_finding(
                WORLD_USER_SURFACE_ENTRY_UNREADABLE,
                subject=defect.name,
                evidence={
                    'index': defect.index,
                    'name': defect.name,
                    'code': defect.code,
                    'value': defect.value,
                    'source': _relative_to_bench(defect.source, cfg) if defect.source is not None else None,
                },
                observed={},
                remedy=f'user surface {defect.name!r} is not captured: its `{defect.code}` ({defect.value}) names no entity type. Fix the '
                'entry in world_state.user_surfaces; the other surfaces are captured.',
                draft_ref=None,
            )
        )
    # THE TURN-INPUT SURFACE, AFTER the world rung on purpose — the error class it raises also covers
    # "the config could not be read at all", whose posture is decided above. Unknown words are findings
    # (an unknown sub-key holds); a malformed value of a known key is invariant.
    try:
        for finding in turn_input_findings(cfg.agent_module_path):
            gates.emit(finding)
    except AgentIOConfigError as exc:
        gates.emit(
            make_finding(
                HARNESS_AGENT_IO_MALFORMED,
                subject='agent.io',
                evidence={'declaration': 'agent.io', 'exc_type': type(exc).__name__},
                observed={'error': str(exc)},
                remedy=f'{exc} The value decides how turn 0 is driven, so it is fixed in the declaration, never guessed.',
                draft_ref=None,
            )
        )
        return _invariant_exit(cfg) or exit_codes.REFUSED
    # THE HOST TOOLCHAIN, in the same pre-spend slot and covering both strategies: a program or module the
    # agent's own files declare, missing or broken where the agent runs it, turns every row that needs it
    # into a grade of the agent's fallback path. A measured gap holds at the funnel; an unmeasured one ships.
    _preflight_host_toolchain(cfg)
    # THE GATE FUNNEL for main()'s pre-spend rungs: a finding a rung above reported stops the run here
    # when it holds — a security or invariant finding, or a holding advisory one no --overrule lifted —
    # and nothing else does. A finding that holds nothing is recorded and never changes the exit code.
    # A gate that runs later (a strategy's own egress preflight) goes through the same funnel after its own emit.
    if (held_code := _held_by_gates(cfg)) is not None:
        return held_code
    if sink is None:
        sink = create_observability_sink(settings)
    if isinstance(sink, ExperimentRunnerSink):
        # THE WORLD, once more: an external world (or an agent-owned one that declares slots) is ONE store
        # every attempt resets and snapshots, so the vendor experiment takes its attempts one at a time
        # whatever --concurrency asked for — or, across N declared world copies, up to min(--concurrency, N)
        # at once, each attempt on its own copy. Its vendor executor cannot wait on a lease without the wait
        # counting against the vendor's task timeout (which requeues), so it keeps this clamp.
        cfg = _one_at_a_time_for_world(cfg, world_owner, world_slots)
        # THE SINGLE-INSTANCE RESOURCE: two agent processes cannot both hold it, so one runs at a time.
        cfg = _one_at_a_time_for_single_instance(cfg, single_instance)
    else:
        # The same world and single-instance resources as LEASES: each held only for an agent process's life and
        # given back before its /finalize, with --concurrency and the trials untouched.
        cfg = _lease_resources(cfg, world_owner, world_slots, single_instance)
    # THE MEMORY: the agent processes alive at once must fit the sandbox's known memory limit (cgroup), or
    # the trials run one after another, each still C wide. No limit known (macOS, no cgroup) → unchanged.
    cfg = _trials_in_turn_for_memory(cfg, _memory_limit_mb())
    code = _run_as_experiment(cfg, settings, sink) if isinstance(sink, ExperimentRunnerSink) else asyncio.run(_run_gathered_async(cfg, settings, sink))
    _report_egress_runtime(cfg.out)
    return code


if __name__ == '__main__':
    sys.exit(main())
