"""Every SDK gate id, declared once.

SDK-OWNED, not vendored: the backend's catalogue declares the backend's
ids, this one declares the ids the SDK's own commands emit, and the two sets are disjoint — except
the scan contract's ids, which BOTH declare from one shared source, the vendored
:mod:`plurai.scan_contract.gate_ids`, because both scan shells mint and overrule them. It keeps the
interface the vendored :mod:`plurai.gates.apply` imports — :func:`gate_spec`, :class:`GateSpec`,
:class:`UnknownGateError` — so ``make_finding`` and ``apply_overrules`` resolve SDK ids here.

A gate id names a gate, not a call site: any number of sites emit the same id, each through its
constant here, never a string literal. :func:`~plurai.gates.apply.make_finding` refuses an id not
declared here, so a misspelled id fails loudly instead of reading as an overrulable advisory.

Each :class:`GateSpec` says:

* ``authority`` — who may lift the finding (:data:`~plurai.gates.contract.GateAuthority`).
* ``holds`` — advisory only: whether the finding holds the command's effect (exit with the
  command's refusal code) until an ``--overrule`` lifts it.
* ``why`` — required for a hard gate and for a holding one: what breaks, or what is held.
* ``evidence_keys`` — exactly the keys a finding's ``evidence`` carries; the fingerprint covers
  them. Stable across runs over the same input: no stderr, paths, timings, errno text or floats
  (volatile context goes to ``observed``).
* ``carriers`` — the commands whose ``--overrule`` can lift it (:data:`~plurai.gates.cli.OVERRULE_COMMANDS`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from plurai.gates.cli import RUNNER_COMMAND, SCAN_CHECK_COMMAND
from plurai.gates.contract import HARD_AUTHORITIES, GateAuthority
from plurai.scan_contract.gate_ids import CLOSED_CLASS_DETECTOR_MISSING, CORPUS_ROOT_EMPTY, FACT_CONTRADICTED, SCAN_GATES


class UnknownGateError(ValueError):
    """A gate id that is not declared in :data:`GATES`."""


@dataclass(frozen=True)
class GateSpec:
    authority: GateAuthority
    holds: bool
    why: str
    evidence_keys: frozenset[str]
    carriers: frozenset[str]

    def __post_init__(self) -> None:
        hard = self.authority in HARD_AUTHORITIES
        if hard and self.holds:
            raise ValueError('a security or invariant gate never holds: no overrule lifts it, so there is no hold to release.')
        if (hard or self.holds) and not self.why.strip():
            raise ValueError('a hard or holding gate must say why.')
        if self.holds and not self.carriers:
            raise ValueError('a holding gate needs at least one carrier, or nothing can ever release its hold.')


def spec(
    authority: GateAuthority,
    *,
    holds: bool,
    why: str,
    evidence_keys: tuple[str, ...],
    carriers: tuple[str, ...],
) -> GateSpec:
    """One catalogue row. Every argument is required, so a row states each property it has."""
    return GateSpec(authority=authority, holds=holds, why=why, evidence_keys=frozenset(evidence_keys), carriers=frozenset(carriers))


#: A scenario's agent was served a stub for one or more tool calls ``/mock/execute`` did not answer
#: (a non-2xx from a hop in front of the mock, a timeout, a refused connection, a non-JSON answer).
#: The scenario is graded; the finding says which calls, what the agent saw, and how to get them
#: answered. World trust is not this gate's: the finalize-time world read decides it.
MOCK_TOOL_CALL_UNANSWERED: Final[str] = 'mock.tool_call_unanswered'

#: A dialog turn failed after the conversation started (the agent raised, hit its recursion budget,
#: its provider failed, or the per-turn wall), and the dialog LANDED — finalized and graded on the
#: turns it reached — instead of being discarded. A record of what cut the dialog, never a hold.
DIALOG_TURN_FAILURE_LANDED: Final[str] = 'dialog.turn_failure_landed'

#: A dialog landed on a PROVIDER failure: infrastructure cut it, so the trial is invalid. Holding: it
#: holds that trial cell (not the run) out of every rate until an overrule says the dialog it reached
#: is worth counting as a measurement.
DIALOG_PROVIDER_INFRA_INVALID: Final[str] = 'dialog.provider_infra_invalid'

#: A dialog landed on the per-turn WALL: the apparatus' budget cut it, so the trial is invalid. Holding, on
#: the same terms as :data:`DIALOG_PROVIDER_INFRA_INVALID`: it holds that trial cell until an overrule says
#: the turns it reached are worth counting.
DIALOG_TURN_BUDGET_INVALID: Final[str] = 'dialog.turn_budget_invalid'

#: The simulated user repeated itself, so the run recorded the dialog ``run_valid: false`` and the trial is
#: invalid. Holding, on the same terms as :data:`DIALOG_PROVIDER_INFRA_INVALID`: it holds that trial cell out of
#: every rate until an overrule says the repeated turn left the measurement standing.
DIALOG_USER_SIM_REPEAT_INVALID: Final[str] = 'dialog.user_sim_repeat_invalid'

#: The holding gates whose held effect is ONE trial cell (kept out of every rate by the trials aggregation, whenever
#: the finding is emitted), never the command: no exit funnel answers for them, so the record lists them as ``held``
#: whether or not a funnel ran before they were emitted.
CELL_HOLDS: Final[frozenset[str]] = frozenset({DIALOG_PROVIDER_INFRA_INVALID, DIALOG_TURN_BUDGET_INVALID, DIALOG_USER_SIM_REPEAT_INVALID})

#: A finalize snapshot channel (the world, the tool trail) could not be read even after a
#: readiness-gated re-read; the dialog was finalized with that channel unread, so the policies that
#: read it abstained.
DIALOG_WORLD_CAPTURE_FAILED: Final[str] = 'dialog.world_capture_failed'

#: The harness's world snapshot carried values with one obvious JSON form (a datetime, a Decimal, a
#: UUID, a set …) and they were carried in that form: a code_eval comparing them reads the JSON form.
WORLD_VALUES_COERCED: Final[str] = 'world.values_coerced'

#: One IPC envelope was longer than the pipe's line limit, so it could not cross: malformed wire.
IPC_LINE_OVERRUN: Final[str] = 'ipc.line_overrun'

#: One or more agent replies were TRIMMED to fit one IPC line: the reply text and actions are the agent's, its message
#: chain and the bulk of its agent state never crossed, so judges reading those turns' tool trail read less than the
#: agent did. The scenario is graded and folds into ``measurement_clean``; the finding names the turns.
IPC_REPLY_TRIMMED: Final[str] = 'ipc.reply_trimmed'

_CONVERT_TRACE: Final[tuple[str, ...]] = ('plurai convert-trace',)

# --- plurai convert-trace: one finding per thread; a hard one skips that thread, never the run ---------------

#: No reader of this SDK reads the convention the artifact is written in.
TRACE_FORMAT_UNSUPPORTED: Final[str] = 'trace.format_unsupported'
#: The artifact cannot be read as the format it was given as.
TRACE_EXPORT_MALFORMED: Final[str] = 'trace.export_malformed'
#: A message, or the pin built from the messages, has no faithful pin shape.
TRACE_MESSAGE_UNREPRESENTABLE: Final[str] = 'trace.message_unrepresentable'
#: The history breaks a resume-seed shape rule the backend enforces.
TRACE_SEED_SHAPE_REFUSED: Final[str] = 'trace.seed_shape_refused'
#: Writing the thread would replace or collide with another folder.
TRACE_FOLDER_CONFLICT: Final[str] = 'trace.folder_conflict'
#: ``--cut-turn`` names no turn of this thread.
TRACE_CUT_OUT_OF_RANGE: Final[str] = 'trace.cut_out_of_range'
#: The env declared ``anonymize: true``; conversion carries content verbatim.
TRACE_POSTURE_ANONYMIZED: Final[str] = 'trace.posture_anonymized'
#: The LangSmith root whose state was pinned is not the only reading of the export.
TRACE_LANGSMITH_ROOT_SELECTION: Final[str] = 'trace.langsmith_root_selection'
#: The conversation was read from a state channel other than ``messages``.
TRACE_MESSAGES_CHANNEL_INFERRED: Final[str] = 'trace.messages_channel_inferred'
#: Calls the source could not parse, and nothing answers, were left out of the pin.
TRACE_INVALID_TOOL_CALLS_DROPPED: Final[str] = 'trace.invalid_tool_calls_dropped'
#: Tool results opening the history, answering no call in it, were left out of the pin.
TRACE_ORPHAN_TOOL_RESULTS_DROPPED: Final[str] = 'trace.orphan_tool_results_dropped'
#: Records no adapter rule accounts for were left out of the conversation.
TRACE_ADAPTER_UNRECOGNISED_RECORDS: Final[str] = 'trace.adapter_unrecognised_records'
#: Model calls off the conversation's lineage, making no tool calls, were left out of the thread.
TRACE_SPAN_SIDE_CALLS_EXCLUDED: Final[str] = 'trace.span_side_calls_excluded'
#: Two model calls are equally the conversation's latest, so which conversation to pin cannot be told.
TRACE_SPAN_LINEAGE_AMBIGUOUS: Final[str] = 'trace.span_lineage_ambiguous'
#: Model calls on the lineage were shown different system messages; the first is carried.
TRACE_SPAN_SYSTEM_CONTEXT_VARIED: Final[str] = 'trace.span_system_context_varied'
#: One artifact holds spans of more than one trace; splitting them is the export's job.
TRACE_ARTIFACT_HOLDS_MANY_TRACES: Final[str] = 'trace.artifact_holds_many_traces'

_PIN_IS_VERBATIM = 'the resume pin carries the conversation verbatim, and the backend refuses at authoring and again at run time a seed it cannot replay'

# ── the runner's egress gates (:mod:`plurai.egress`, ``run_simulation._preflight_egress``) ──────────────
EGRESS_GUARD_BLOCKED: Final[str] = 'egress.guard_blocked'
EGRESS_BREACHED_UNENFORCED: Final[str] = 'egress.breached_unenforced'
EGRESS_BREACHED: Final[str] = 'egress.breached'
EGRESS_UNPROVABLE: Final[str] = 'egress.unprovable'
EGRESS_LICENSED_ONLY_WITHHELD: Final[str] = 'egress.licensed_only_withheld'
EGRESS_DECLARATION_ENTRY_UNUSABLE: Final[str] = 'egress.declaration_entry_unusable'
EGRESS_DECLARATION_UNUSABLE: Final[str] = 'egress.declaration_unusable'
EGRESS_PREFLIGHT_CRASHED: Final[str] = 'egress.preflight_crashed'
EGRESS_GUARD_UNCONFIRMED: Final[str] = 'egress.guard_unconfirmed'
EGRESS_GUARD_COVERAGE_GAP: Final[str] = 'egress.guard_coverage_gap'
EGRESS_PROBE_MODE_REDUCED: Final[str] = 'egress.probe_mode_reduced'
EGRESS_DENY_LIST_INCOMPLETE: Final[str] = 'egress.deny_list_incomplete'
EGRESS_UNGUARDED_UNMEASURED: Final[str] = 'egress.unguarded_unmeasured'

# ── the runner's pre-spend rungs (``run_simulation._main_body``) ─────────────────────────────────────────
PREFLIGHT_BACKEND_UNREACHABLE: Final[str] = 'preflight.backend_unreachable'
PREFLIGHT_BACKEND_REDIRECTED: Final[str] = 'preflight.backend_redirected'
PREFLIGHT_BACKEND_UNHEALTHY: Final[str] = 'preflight.backend_unhealthy'
PREFLIGHT_BACKEND_REJECTS_CREDENTIALS: Final[str] = 'preflight.backend_rejects_credentials'
PREFLIGHT_BACKEND_AUTH_FAILED: Final[str] = 'preflight.backend_auth_failed'
PREFLIGHT_MOCK_UNREACHABLE: Final[str] = 'preflight.mock_unreachable'
PREFLIGHT_MOCK_URL_UNUSABLE: Final[str] = 'preflight.mock_url_unusable'
PREFLIGHT_MOCK_REJECTS_CREDENTIALS: Final[str] = 'preflight.mock_rejects_credentials'
RUN_OUT_OCCUPIED: Final[str] = 'run.out_occupied'
CONFIG_UNREADABLE: Final[str] = 'config.unreadable'
HARNESS_PIPELINE_UNDECLARED: Final[str] = 'harness.pipeline_undeclared'
HARNESS_SHAPE_OVERRIDE_CONTRADICTS_DECLARATION: Final[str] = 'harness.shape_override_contradicts_declaration'
HARNESS_AGENT_IO_MALFORMED: Final[str] = 'harness.agent_io_malformed'
HARNESS_AGENT_IO_UNKNOWN_FIELD: Final[str] = 'harness.agent_io_unknown_field'
HARNESS_AGENT_IO_UNKNOWN_PART_KIND: Final[str] = 'harness.agent_io_unknown_part_kind'
WORLD_UNRESET_EXTERNAL: Final[str] = 'world.unreset_external'
WORLD_USER_SURFACES_MALFORMED: Final[str] = 'world.user_surfaces_malformed'
WORLD_USER_SURFACE_ENTRY_UNREADABLE: Final[str] = 'world.user_surface_entry_unreadable'
HOST_REQUIREMENT_GAP: Final[str] = 'host.requirement_gap'
HOST_REQUIREMENT_UNVERIFIED: Final[str] = 'host.requirement_unverified'
HOST_REQUIREMENT_NOT_CHECKED: Final[str] = 'host.requirement_not_checked'

_RUNNER: Final[tuple[str, ...]] = (RUNNER_COMMAND,)
_BREACH_EVIDENCE: Final[tuple[str, ...]] = ('host', 'reachable_ports', 'source', 'agent_runtime', 'enforcement', 'harness_suffix', 'mock_mode')
_HOST_EVIDENCE: Final[tuple[str, ...]] = ('name', 'kind', 'status', 'confinement', 'probe')


def _advisory(*evidence_keys: str) -> GateSpec:
    """A finding that holds nothing: it rides the record and never touches the exit code."""
    return spec('advisory', holds=False, why='', evidence_keys=evidence_keys, carriers=_RUNNER)


def _holding(why: str, *evidence_keys: str) -> GateSpec:
    """An advisory finding that holds the run (exit 2, nothing spent) until an ``--overrule`` lifts it."""
    return spec('advisory', holds=True, why=why, evidence_keys=evidence_keys, carriers=_RUNNER)


def _invariant(why: str, *evidence_keys: str) -> GateSpec:
    """Nothing downstream could run or read what this gate saw; the remedy is a repair, never an overrule."""
    return spec('invariant', holds=False, why=why, evidence_keys=evidence_keys, carriers=_RUNNER)


# ── the scan contract (``plurai-scan-contract check``, ``run_simulation._preflight_scan_contract``) ─────────
# FACT_CONTRADICTED, CLOSED_CLASS_DETECTOR_MISSING and CORPUS_ROOT_EMPTY are SHARED with the backend: declared
# once in the vendored ``plurai.scan_contract.gate_ids`` and applied inside the vendored checker. The
# run-time drift gate is the SDK's own.

#: A full run whose scanned corpus MEASURABLY changed since the baseline the suite was built over.
RUN_SCAN_SOURCE_DRIFT: Final[str] = 'run.scan_source_drift'

#: A full run was given a ``--scan-contract`` baseline and the drift comparison could not be MADE (no or a missing
#: ``--scan-corpus`` root, an unusable baseline, a corpus that could not be re-hashed, a scope the comparison abstains
#: on, or two sides sharing no path — a wrong root or a wholesale swap). Not a drift verdict: the evidence is the
#: typed reason the comparison did not happen, one of ``no_corpus_root``, ``corpus_root_missing``,
#: ``baseline_unusable``, ``corpus_unhashable``, ``comparison_blocked``, ``shares_no_path``, ``recheck_raised``.
RUN_SCAN_DRIFT_UNCHECKED: Final[str] = 'run.scan_drift_unchecked'

_SCAN_ROWS: Final[dict[str, GateSpec]] = {
    gate_id: spec('advisory', holds=True, why=SCAN_GATES[gate_id].why, evidence_keys=SCAN_GATES[gate_id].evidence_keys, carriers=(SCAN_CHECK_COMMAND,))
    for gate_id in (FACT_CONTRADICTED, CLOSED_CLASS_DETECTOR_MISSING, CORPUS_ROOT_EMPTY)
}

# ── the code action channel's interception (``plurai._mocking.interception_watch``) ───────────────────────

#: A file routing the agent's vendor calls to the mock changed during the scenario, so calls after the change
#: may have reached the real service and the world holds none of their effects. The scenario is graded; the
#: finding says which files.
MOCK_INTERCEPTION_CHANGED: Final[str] = 'mock.interception_changed'

_INTERCEPTION_ROWS: Final[dict[str, GateSpec]] = {MOCK_INTERCEPTION_CHANGED: _advisory('files')}

#: THE catalogue. Read-only. Each gate id is a ``Final[str]`` constant in this module, keyed here to its row.
GATES: Final[Mapping[str, GateSpec]] = MappingProxyType(
    {
        MOCK_TOOL_CALL_UNANSWERED: spec(
            'advisory',
            holds=False,
            why='',
            evidence_keys=('calls', 'calls_unanswered', 'restart_shaped', 'served_schema_stub'),
            carriers=(),
        ),
        DIALOG_TURN_FAILURE_LANDED: spec(
            'advisory', holds=False, why='', evidence_keys=('reason', 'kind', 'turn_idx', 'closing', 'child_exc_class'), carriers=()
        ),
        DIALOG_PROVIDER_INFRA_INVALID: spec(
            'advisory',
            holds=True,
            why='holds the trial cell of a dialog the agent provider cut out of every rate; an overrule counts the turns it reached',
            evidence_keys=('turn_idx', 'closing', 'child_exc_class'),
            carriers=(RUNNER_COMMAND,),
        ),
        DIALOG_TURN_BUDGET_INVALID: spec(
            'advisory',
            holds=True,
            why='holds the trial cell of a dialog the per-turn wall cut out of every rate; an overrule counts the turns it reached',
            evidence_keys=('turn_idx', 'closing', 'child_exc_class'),
            carriers=(RUNNER_COMMAND,),
        ),
        DIALOG_USER_SIM_REPEAT_INVALID: spec(
            'advisory',
            holds=True,
            why='holds the trial cell of a dialog whose simulated user repeated itself out of every rate; an overrule counts it',
            evidence_keys=('pairs', 'owners'),
            carriers=(RUNNER_COMMAND,),
        ),
        DIALOG_WORLD_CAPTURE_FAILED: spec('advisory', holds=False, why='', evidence_keys=('channels',), carriers=()),
        WORLD_VALUES_COERCED: spec('advisory', holds=False, why='', evidence_keys=('paths',), carriers=()),
        IPC_REPLY_TRIMMED: spec('advisory', holds=False, why='', evidence_keys=('turns',), carriers=()),
        IPC_LINE_OVERRUN: spec(
            'invariant',
            holds=False,
            why='an envelope longer than the IPC line limit cannot be read by the other side; the attempt has no reply to grade',
            evidence_keys=('envelope_type', 'direction', 'limit'),
            carriers=(),
        ),
        TRACE_FORMAT_UNSUPPORTED: spec(
            'invariant',
            holds=False,
            why=f'a reading by resemblance would mislabel speakers; {_PIN_IS_VERBATIM}',
            evidence_keys=('source_format', 'convention'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_EXPORT_MALFORMED: spec(
            'invariant',
            holds=False,
            why=f'the artifact does not hold the conversation in a shape this reader can carry; {_PIN_IS_VERBATIM}',
            evidence_keys=('source_format', 'step'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_MESSAGE_UNREPRESENTABLE: spec(
            'invariant',
            holds=False,
            why=f'a message the pin schema cannot hold would be lost or invented; {_PIN_IS_VERBATIM}',
            evidence_keys=('source_format', 'step'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_SEED_SHAPE_REFUSED: spec(
            'invariant',
            holds=False,
            why='the backend refuses this seed shape at authoring and at run time, and a provider refuses an unpaired tool call on turn 1',
            evidence_keys=('rule_id', 'rules_version', 'message_index'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_FOLDER_CONFLICT: spec(
            'invariant',
            holds=False,
            why='writing would overwrite existing output, or two threads would land in one folder',
            evidence_keys=('folder', 'kind'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_CUT_OUT_OF_RANGE: spec(
            'invariant',
            holds=False,
            why='the cut names no turn this thread has, so there is no opener to lift and no history to seed',
            evidence_keys=('cut_turn', 'user_turns'),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_SPAN_LINEAGE_AMBIGUOUS: spec(
            'invariant',
            holds=False,
            why=f'pinning either call would drop the other as a side call, and nothing in the trace says which is the conversation; {_PIN_IS_VERBATIM}',
            evidence_keys=('spans',),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_POSTURE_ANONYMIZED: spec(
            'security',
            holds=False,
            why='the env has not consented to verbatim content leaving it, and a resume pin is verbatim by construction',
            evidence_keys=('declared_by',),
            carriers=_CONVERT_TRACE,
        ),
        TRACE_LANGSMITH_ROOT_SELECTION: spec(
            'advisory', holds=False, why='', evidence_keys=('chosen', 'skipped_stateless', 'disjoint_roots', 'longest'), carriers=_CONVERT_TRACE
        ),
        TRACE_MESSAGES_CHANNEL_INFERRED: spec('advisory', holds=False, why='', evidence_keys=('channel',), carriers=_CONVERT_TRACE),
        TRACE_INVALID_TOOL_CALLS_DROPPED: spec('advisory', holds=False, why='', evidence_keys=('messages',), carriers=_CONVERT_TRACE),
        TRACE_ORPHAN_TOOL_RESULTS_DROPPED: spec('advisory', holds=False, why='', evidence_keys=('tool_call_ids',), carriers=_CONVERT_TRACE),
        TRACE_ADAPTER_UNRECOGNISED_RECORDS: spec('advisory', holds=False, why='', evidence_keys=('count', 'shapes'), carriers=_CONVERT_TRACE),
        TRACE_SPAN_SIDE_CALLS_EXCLUDED: spec('advisory', holds=False, why='', evidence_keys=('spans',), carriers=_CONVERT_TRACE),
        TRACE_SPAN_SYSTEM_CONTEXT_VARIED: spec('advisory', holds=False, why='', evidence_keys=('spans',), carriers=_CONVERT_TRACE),
        TRACE_ARTIFACT_HOLDS_MANY_TRACES: spec(
            'invariant',
            holds=False,
            why=f'which trace is the conversation is not in the artifact, and reading them as one would merge two conversations; {_PIN_IS_VERBATIM}',
            evidence_keys=('source_format', 'trace_ids'),
            carriers=_CONVERT_TRACE,
        ),
        EGRESS_GUARD_BLOCKED: spec(
            'security',
            holds=False,
            why='the in-process guard refused a connection to a declared production host: that refusal IS the isolation. '
            'Consent to reach a host live is `egress.allow_hosts` in the committed config, never an overrule.',
            evidence_keys=('host', 'callsite', 'scope'),
            carriers=_RUNNER,
        ),
        EGRESS_BREACHED_UNENFORCED: _holding(
            'a declared production host is reachable from this machine and nothing enforces isolation for this agent '
            '(no in-process guard, no hermetic tool channel): a simulated user would drive real actions before anyone read the finding',
            *_BREACH_EVIDENCE,
        ),
        EGRESS_BREACHED: _advisory(*_BREACH_EVIDENCE),
        EGRESS_UNPROVABLE: _advisory('unproven', 'guessed_ports', 'env_hosts', 'cause'),
        EGRESS_LICENSED_ONLY_WITHHELD: _advisory('reasons'),
        EGRESS_DECLARATION_ENTRY_UNUSABLE: _advisory('key', 'index', 'code', 'value'),
        EGRESS_DECLARATION_UNUSABLE: _advisory('key', 'code'),
        EGRESS_PREFLIGHT_CRASHED: _advisory('exc_type', 'phase'),
        EGRESS_GUARD_UNCONFIRMED: _advisory('child_python', 'required_plurai'),
        EGRESS_GUARD_COVERAGE_GAP: _advisory('gap', 'commands'),
        EGRESS_PROBE_MODE_REDUCED: _advisory('probe_mode', 'hosts'),
        EGRESS_DENY_LIST_INCOMPLETE: _holding(
            'the `egress` block could not be read where it names production hosts (a YAML error, a non-mapping block, a non-list '
            '`production_hosts`, or an entry that is not a host), so the in-process guard would run missing hosts the env declared',
            'key',
            'index',
            'code',
            'value',
        ),
        EGRESS_UNGUARDED_UNMEASURED: _holding(
            'the egress preflight raised where nothing else stands in for it (no guard armed from the declaration, a probe that '
            'crashed for an agent no enforcement layer covers, or a child that never confirmed the guard and a re-probe that '
            'crashed): the agent would run neither guarded nor measured',
            'phase',
            'exc_type',
        ),
        PREFLIGHT_BACKEND_UNREACHABLE: _invariant('the backend URL does not resolve, so no thread could be opened', 'base_url', 'attempts'),
        PREFLIGHT_BACKEND_REDIRECTED: _invariant(
            'the backend answered with a redirect, and the run client follows none, so every call would get the same answer',
            'base_url',
            'status',
            'location',
        ),
        PREFLIGHT_BACKEND_UNHEALTHY: _holding(
            'the backend health route answered non-2xx: it may be a gateway in front of a working API, or no backend at all, '
            'and nothing is spent (no dataset upload, no thread) until that is decided',
            'base_url',
            'path',
            'status',
        ),
        PREFLIGHT_BACKEND_REJECTS_CREDENTIALS: _invariant(
            "a bearer-gated backend route answered 401 to this run's own credentials, so every call of the run would be refused",
            'base_url',
            'token_sent',
        ),
        PREFLIGHT_BACKEND_AUTH_FAILED: _holding(
            "the backend health route answered 401 or 403 to this run's own headers while the bearer-gated route did not "
            'answer 401: an auth layer in front of the box refuses the run, or gates only that route, and nothing is spent '
            'until that is decided',
            'base_url',
            'path',
            'status',
            'token_sent',
        ),
        PREFLIGHT_MOCK_UNREACHABLE: _invariant(
            'the mock url this invocation pinned refused every connection; the run would score a world nobody deployed', 'mock_url', 'pin_source'
        ),
        PREFLIGHT_MOCK_URL_UNUSABLE: _invariant(
            'the pinned mock url cannot be dialed at all; every scenario would die on the same error', 'mock_url', 'pin_source', 'exc_type'
        ),
        PREFLIGHT_MOCK_REJECTS_CREDENTIALS: _invariant(
            "the pinned mock answered 401 to this run's credentials, so every tool call and world read would be refused", 'mock_url', 'token_sent'
        ),
        RUN_OUT_OCCUPIED: _invariant('--out holds a finished run whose summary this run would delete', 'partial', 'replace'),
        CONFIG_UNREADABLE: _invariant('every reader of plurai.config.yaml downstream raises on the same file', 'config', 'exc_type'),
        HARNESS_PIPELINE_UNDECLARED: _invariant(
            'the harness exports only invoke_pipeline and declares no shape, so every scenario would die on turn 1', 'harness', 'hooks'
        ),
        HARNESS_SHAPE_OVERRIDE_CONTRADICTS_DECLARATION: _advisory('declared', 'override', 'config'),
        HARNESS_AGENT_IO_MALFORMED: _invariant(
            'a known agent.io key has a value that decides how turn 0 is driven and cannot be read; guessing drives the harness against its own declaration',
            'declaration',
            'exc_type',
        ),
        HARNESS_AGENT_IO_UNKNOWN_FIELD: _holding(
            'agent.io carries a key this build does not read; a misspelled turn-0 key would drive every scenario the wrong way',
            'key',
            'value',
            'source',
        ),
        HARNESS_AGENT_IO_UNKNOWN_PART_KIND: _advisory('key', 'value', 'source'),
        WORLD_UNRESET_EXTERNAL: _holding(
            'the declared world is not the mock and no module-scope reset is bound, so each scenario may open on its '
            "predecessor's writes; only a reader of the agent's tools can say whether it writes",
            'owner',
            'config',
            'harness',
            'reset_bound',
            'unreadable_as',
        ),
        WORLD_USER_SURFACES_MALFORMED: _invariant(
            'world_state.user_surfaces cannot be parsed as a list of surfaces, so no capture could be scoped', 'declaration', 'exc_type'
        ),
        WORLD_USER_SURFACE_ENTRY_UNREADABLE: _advisory('index', 'name', 'code', 'value', 'source'),
        HOST_REQUIREMENT_GAP: _holding(
            "a program or module the agent's own files declare is missing, broken or fails where the agent runs it, so the "
            "rows that need it grade the agent's fallback path",
            *_HOST_EVIDENCE,
        ),
        HOST_REQUIREMENT_UNVERIFIED: _advisory(*_HOST_EVIDENCE),
        HOST_REQUIREMENT_NOT_CHECKED: _advisory(*_HOST_EVIDENCE),
        **_SCAN_ROWS,
        **_INTERCEPTION_ROWS,
        RUN_SCAN_SOURCE_DRIFT: spec(
            'advisory',
            holds=True,
            why='a final scored run over a corpus that changed since the scan scores the agent against premises the suite no longer '
            'describes; the run waits for a rebuild, a fresh baseline, or an overrule',
            evidence_keys=('recorded_hash', 'current_hash', 'added', 'removed', 'changed'),
            carriers=(RUNNER_COMMAND,),
        ),
        RUN_SCAN_DRIFT_UNCHECKED: spec(
            'advisory',
            holds=True,
            why='a final scored run was asked to verify its corpus against a baseline and could not, so nothing says the suite '
            'still describes the sources the agent reads; the run waits for a usable baseline and roots, or an overrule',
            evidence_keys=('reason_code',),
            carriers=(RUNNER_COMMAND,),
        ),
    }
)


def gate_spec(gate_id: str) -> GateSpec:
    """The catalogue entry for ``gate_id``; raises :class:`UnknownGateError` for an undeclared id."""
    found = GATES.get(gate_id)
    if found is None:
        raise UnknownGateError(f'gate id {gate_id!r} is not declared in plurai.gates.catalogue.GATES.')
    return found


def is_sdk_gate(gate_id: str) -> bool:
    """Whether the SDK owns ``gate_id`` — so an overrule naming it is applied here rather than forwarded."""
    return gate_id in GATES
