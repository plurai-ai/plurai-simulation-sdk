"""``plurai-scan-contract`` — run the scan's own conclusions against the corpus they came from.

The checker itself is a copy of the backend's checker (``plurai/scan_contract/``, kept
byte-identical); this module is the SDK-owned shell around
it. The backend ships its own scan-contract CLI, but customers install only the SDK and the
plugin, so the check has to run from the SDK. Two constraints decide that, and both are properties
of WHERE the sources live:

* the SDK cannot depend on the backend package — it would drag in the whole backend runtime and
  private dependencies a customer cannot resolve;
* the backend cannot run the check — it would verify the uploaded snapshot, which by
  construction never drifts from itself. The drift check only means anything where the files
  are, and that is the customer's machine.

Two commands, with the flags of the backend's own scan CLI. A subcommand is REQUIRED; a bare call
prints usage and exits ``USAGE``.

``check``
    Derive + verify + report. Exits ``SCAN_HELD`` while a HOLDING finding is un-overruled — a typed fact the
    corpus refutes, a closed set whose required detector none passed, or a corpus root that
    admitted no file — so a scan step wired to it stops before generation spends. ``--overrule
    FILE`` (:mod:`plurai.gates.cli`) lifts a hold the caller has judged, bound to the finding's
    fingerprint; the vendored checker applies it, and the verdict underneath never changes. A
    ``scan.*`` overrule id no scan gate declares is a usage error. What each overrule did is
    recorded INTERNALLY under ``<parent of --out>/.plurai/gate_authority/`` (:func:`gate_record_path`)
    and counted on one ``[gate]`` stderr line. Everything else the report says is advisory and never
    moves the exit code.

``check`` also CARRIES the agent's declared render / action surface. The
``render_actions:`` list a ``plurai.config.yaml`` under a ``--source-root`` declares
(:mod:`plurai.render_actions`; the plugin's scan step passes the repo root and the ``plurai_sim``
workdir, so the config is under a root it already gives) is written to ``scan_contract.json`` as
a top-level ``render_actions`` field — ``{"verbs": [...], "source": <config path>}`` — and the
field is ABSENT when no config declares one; ``render_actions: []`` carries an empty ``verbs``,
which is a declaration. A malformed declaration is DROPPED with an advisory report item naming
it, and the scan runs without it: an undeclared surface means "nothing to compare", which never
false-reds. The field then carries ``{"unreadable": <why>}`` (the vendored ``artifact_payload``
writes it), so the backend reports the declaration as present and unreadable, not as absent.
This is the one field of the artifact the SDK shell owns: ``artifact_payload`` is vendored, and the
backend's own CLI reads no config, so it writes no such field;
the backend's ``RENDER_ACTIONS`` fact check reads the field to compare an invented verb against
the declared surface.

``recheck``
    Compare the sources as they are NOW against the inventory a previous ``check`` recorded, and
    report match/mismatch. Exits 0 on any VERDICT: source drift is advisory (see
    ``plurai.scan_contract.corpus.check_source_drift``) and a run must not be blocked by it.
    :func:`recheck_source_drift` is the same comparison as a function — the run-start preflight
    in ``plurai.run_simulation`` calls it, so the check has a caller in the flow and not only on
    a command line. It re-walks at the scope the baseline RECORDED, so a baseline taken at another
    scope keeps its drift measured.

FORMATS: both commands read every format :mod:`plurai.readers` reads, through
:data:`plurai.readers.SCAN_READER` plugged into the vendored checker's ``SourceReader`` seam. The
stored bytes are hashed (a ``.gz`` compressed, a zip member's own bytes), and the converted text is
what facts are decided over. A suffix with no reader is named in the report as not read; a binary
admitted by ``--suffixes`` is refused by name, never searched as text.

EXIT CODES are the contract, because the plugin's scan step branches on them and never on the
markdown, and they are rows of :mod:`plurai.exit_codes`: ``0`` no held finding, ``SCAN_HELD`` (18) a HELD
finding, ``USAGE`` (64) a usage error or a malformed input document (including a malformed ``--overrule``
file), ``REFUSED`` (65) an unsupported platform, ``INTERNAL_FAULT`` (70) THE CHECKER ITSELF FAILED — see
:func:`main`.

``check`` CAN REACH THE LIVE AGENT — the only part of this CLI that leaves the machine's own
files, and it does so only when BOTH ``--probe-consent`` and ``--agent-command`` are given.
Neither alone fires anything: without consent the derived probes are recorded not-run
with that reason, and without a command there is no transport to fire them through. What is sent
is one derived ask per probe, over the customer's OWN command (stdin → stdout), under a hard
``--smoke-budget`` cap and a per-probe ``--probe-timeout``; an ask whose clause opens on an
imperative state-changing verb is REFUSED before it is sent, and the refusal is recorded. The
replies are transcripts and land in the report as evidence, so treat the artifact as carrying
agent output. With neither flag this command touches nothing but the corpus directories.

Typer and rich are deliberately absent: this runs on a customer machine, and the console script
must not add dependencies to an SDK install that a customer never asked for.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plurai import exit_codes
from plurai import readers
from plurai._platform import utf8_stdio
from plurai.gates.catalogue import UnknownGateError
from plurai.gates.cli import OverruleFileError, add_overrule_argument, load_overrules
from plurai.ingest.anonymize import KEY_DIR_NAME
from plurai.ingest.cli import GATE_RECORD_DIRNAME, _prepare_gate_record_dir  # noqa: PLC2701 - the ingest commands' own self-ignoring record dir, reused
from plurai.render_actions import RENDER_ACTIONS_KEY, RenderActionsConfigError, discover_render_surface
from plurai.scan_contract.closed_class_gate import CONFIG_KEY as CLOSED_CLASS_CONFIG_KEY
from plurai.scan_contract.closed_class_gate import discover_closed_class_overrides
from plurai.scan_contract.contract import (
    ScanContractResult,
    artifact_payload,
    fact_dependents,
    gate_authority_payload,
    run_scan_contract,
)
from plurai.scan_contract.corpus import (
    DriftReport,
    SourceInventory,
    build_source_inventory,
    check_source_drift,
    load_corpus_roots,
    normalize_suffixes,
)
from plurai.scan_contract.facts import load_scan_facts, verify_facts
from plurai.scan_contract.probes import DEFAULT_PROBE_BUDGET
from plurai.scan_contract.report import DeclarationProblem
from plurai.scan_contract.services import load_service_indicators
from plurai.scan_contract.smoke_execution import DEFAULT_PROBE_TIMEOUT, execute_smoke_probes, record_execution_in_report, subprocess_transport
from plurai.scan_contract.smoke_probes import DEFAULT_SMOKE_BUDGET, SmokeProbeSet
from plurai.user_path import expand_user_path

#: The two artifacts, named exactly as the backend names them: ``recheck --against`` reads the first,
#: and the plugin's scan step reads both by these names.
ARTIFACT_FILENAME = 'scan_contract.json'
REPORT_FILENAME = 'pre_generation_findings.md'

#: ``check``'s INTERNAL gate record (findings + every overrule outcome, overrule reasons included),
#: named the way an ingest command's is: ``<workspace>/.plurai/gate_authority/<command>.json``.
GATE_RECORD_FILENAME = 'scan-contract-check.json'


def gate_record_path(out: Path) -> Path:
    """Where ``check --out OUT`` writes its gate record: under the private, self-ignoring ``.plurai/`` of
    the workspace that holds ``OUT`` — never in ``OUT`` itself, beside the ``scan_contract.json`` that is
    committed and uploaded. The plugin passes ``--out <workdir>/artifacts``, so this is the same
    ``<workdir>/.plurai/gate_authority/`` its ingest commands (``--workspace-root <workdir>``) write to."""
    return out.parent / KEY_DIR_NAME / GATE_RECORD_DIRNAME / GATE_RECORD_FILENAME


class BadUsageError(ValueError):
    """A usage error the user can fix — reported as one line and exit ``USAGE``, never a traceback."""


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``path`` atomically.

    ``scan_contract.json`` is the drift BASELINE a later ``recheck --against`` reads, and a torn
    write does not fail loudly there: a tear landing after ``content_hash`` yields a short
    ``files`` list, and every dropped file then reads as ``removed`` in the next DriftReport.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory tmp + replace, and the tmp name is unique PER WRITER so two writers cannot
    # rename each other's half-written file away. NOT fsynced and NOT mkstemp: the 0644 default
    # mode is what a later reader (a different user, a CI step) needs — mkstemp's 0600 would make
    # the baseline unreadable to them.
    tmp = path.with_name(f'.{path.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp')
    try:
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


@dataclass(frozen=True, slots=True)
class RecheckOutcome:
    """A drift comparison and whether it could be MADE.

    ``drift is None`` is the third state and the one that must never read as a match: the
    baseline was absent or unusable, so nothing was compared. Both the CLI and the run-start
    preflight branch on this rather than on ``drift.matched``, which does not exist there.
    """

    drift: DriftReport | None
    baseline: Path | None
    reason: str | None = None
    #: How many files each side of the comparison held. Carried because a DriftReport names the
    #: added/removed/changed PATHS but not the population, and "the two sides share no path at
    #: all" — the signature of a scope mismatch rather than of drift — is only computable with
    #: the recorded count in hand. See :attr:`shares_no_path`.
    recorded_files: int = 0
    current_files: int = 0
    #: The suffix scope the sources were re-walked at: the baseline's recorded scope when it has one,
    #: else the caller's. Empty when the baseline could not be read. ``recheck --facts`` re-reads the
    #: sources at this same scope, so the facts are re-verified over the files the check decided them over.
    scope: tuple[str, ...] = ()

    @property
    def checked(self) -> bool:
        return self.drift is not None

    @property
    def shares_no_path(self) -> bool:
        """Not one recorded path survived into the current inventory.

        Two different situations produce this and NEITHER is ordinary drift: the roots handed to
        the re-check are not the roots that were recorded (a different directory, or the
        multi-root path prefix that ``build_source_inventory`` applies only when it is given more
        than one root), or every single source was replaced. From here they are indistinguishable,
        so this is reported as a comparison that could not be made — never as a drift verdict a
        caller might act on.
        """
        return self.drift is not None and not self.drift.matched and self.recorded_files > 0 and len(self.drift.removed) == self.recorded_files

    @property
    def matched(self) -> bool:
        """True ONLY on a comparison that was made and agreed. Never on one that was skipped."""
        return self.drift is not None and self.drift.matched

    def summary(self) -> str:
        if self.drift is None:
            return f'corpus source drift NOT checked: {self.reason or "no baseline"}'
        return self.drift.summary()

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            'checked': self.checked,
            'summary': self.summary(),
            'baseline': str(self.baseline) if self.baseline else None,
            'recorded_files': self.recorded_files,
            'current_files': self.current_files,
        }
        if self.drift is not None:
            record['drift'] = self.drift.as_dict()
        elif self.reason:
            record['reason'] = self.reason
        return record


def read_recorded_inventory(path: Path) -> SourceInventory:
    """The baseline inventory a previous ``check`` recorded in its ``scan_contract.json``.

    Raises :class:`BadUsageError` rather than returning ``None``: a caller that PASSED a baseline
    asked for a comparison, and answering "no comparison was requested" tells them the opposite
    of what happened.
    """
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise BadUsageError(f'--against {path}: not readable as JSON ({exc}).') from exc
    inventory = payload.get('source_inventory') if isinstance(payload, dict) else None
    if not isinstance(inventory, dict) or not inventory.get('content_hash'):
        raise BadUsageError(
            f'--against {path}: carries no usable source_inventory.content_hash, so it cannot be a drift baseline. '
            'Point at a scan_contract.json written by a previous `check`.'
        )
    recorded = SourceInventory.from_dict(inventory)
    if not recorded.files:
        # A hash with no per-file digests is not a baseline, it is half of one — and the half a
        # comparison needs. Every current file reports as `added`, nothing as `removed`, so the
        # "the two sides share no path" guard cannot fire (it needs a recorded population) and
        # the result renders as a confident DRIFT verdict over a comparison that never had a
        # left-hand side.
        raise BadUsageError(
            f'--against {path}: source_inventory carries a content_hash but no `files`, so there is nothing to compare '
            'per file. Re-record the baseline with `plurai-scan-contract check`.'
        )
    return recorded


def recheck_source_drift(corpus_roots: list[Path], baseline: Path, *, suffixes: Sequence[str] = readers.SCAN_SUFFIXES) -> RecheckOutcome:
    """Re-hash ``corpus_roots`` and compare them against ``baseline``'s recorded inventory.

    THE run-start drift re-check, as a function so it has callers other than a command line: the
    runner's preflight calls this before a scored run.

    Hashes ONLY. ``load_corpus_roots`` would additionally YAML-parse and flatten every structured
    source — measured upstream at ~170x the cost of the hashing this needs, and it reads none of
    it.

    Never raises for an unusable baseline or an unreadable root: the outcome records that the
    comparison did not happen, which is a different fact from "the sources match" and must stay
    different all the way to the artifact.

    The sources are re-walked at the scope the BASELINE recorded (``source_inventory.suffixes``),
    and ``suffixes`` is used only for a baseline that recorded none. Both sides must be walked at
    one scope: a baseline recorded with ``--suffixes html`` and re-hashed without it would report
    every one of those files as ``removed`` with not one of them touched on disk — a drift verdict
    manufactured by the scope of the second walk. ``check_source_drift`` abstains on such a
    mismatch, but abstaining is not measuring: a baseline recorded at another scope would go
    unmeasured. Walking at the recorded scope keeps it measured, with no flag to keep in step.
    """
    try:
        recorded = read_recorded_inventory(baseline)
    except BadUsageError as exc:
        return RecheckOutcome(drift=None, baseline=baseline, reason=str(exc))
    scope = recorded.suffixes or normalize_suffixes(suffixes)
    try:
        current = build_source_inventory(corpus_roots, suffixes=scope, reader=readers.SCAN_READER)
    except Exception as exc:  # noqa: BLE001 — see below: this primitive may not raise
        # ``Exception``, not ``OSError``. The CLI command over this is documented "always exits
        # 0", and the plugin BRANCHES on that: any other exit is read as FIX-FIRST or NOT-RUN. A
        # UnicodeDecodeError or a surprise from the walk would traceback out through
        # ``main`` as exit 1 — an advisory check reporting itself as a blocking finding.
        return RecheckOutcome(drift=None, baseline=baseline, reason=f'the corpus roots could not be re-hashed ({type(exc).__name__}: {exc})', scope=scope)
    drift = check_source_drift(recorded, current)
    if drift.blocked_reason:
        # A BLOCKED report is a comparison that did not happen, and this class already has a
        # state for that — ``drift is None``, which ``checked`` and the preflight both branch on.
        # Handing the report through as-is would publish ``checked: True`` beside a
        # ``matched: False`` nobody measured, which is the reading this whole path exists to stop.
        return RecheckOutcome(
            drift=None,
            baseline=baseline,
            reason=drift.blocked_reason,
            recorded_files=len(recorded.files),
            current_files=len(current.files),
            scope=scope,
        )
    return RecheckOutcome(
        drift=drift,
        baseline=baseline,
        recorded_files=len(recorded.files),
        current_files=len(current.files),
        scope=scope,
    )


def admitted_suffixes(extra: Sequence[str] | None) -> tuple[str, ...]:
    """Every format the SDK reads (:data:`plurai.readers.SCAN_SUFFIXES`) WIDENED by ``extra``, never replaced by it.

    The default is every readable format, read through :data:`plurai.readers.SCAN_READER`: a
    ``.docx``, ``.pdf`` or ``.json.gz`` is in scope without a flag, and its converted text is what
    the facts are decided over. ``extra`` is for PLAIN TEXT in an extension no reader names (``ts``,
    ``py``), which the reader decodes as text — a binary is refused by name, never searched.

    Additive because the silent direction is the narrow one: a scan that reads fewer files than
    the operator thinks reports exactly like one that read them all. ``normalize_suffixes``
    accepts ``html`` and ``.HTML`` alike, so a flag never parses, runs and admits nothing.

    PUBLIC because three entry points widen the same way and must agree on the tuple: this CLI's
    ``--suffixes``, the runner's ``--scan-suffixes`` (the re-check scope of a baseline that recorded
    none — :func:`recheck_source_drift`) and the corpus ingest's ``--suffixes``.
    """
    return normalize_suffixes((*readers.SCAN_SUFFIXES, *(extra or ())))


def _admitted_suffixes(args: argparse.Namespace) -> tuple[str, ...]:
    """:func:`admitted_suffixes` over an argparse namespace."""
    return admitted_suffixes(args.suffixes)


def _cmd_check(args: argparse.Namespace) -> int:
    corpus: list[Path] = args.corpus
    try:
        overrules = load_overrules(args.overrule).overrules if args.overrule else ()
    except OverruleFileError as exc:
        raise BadUsageError(f'--overrule: {exc}') from exc
    try:
        units, inventory, warnings, empty_roots = load_corpus_roots(corpus, suffixes=_admitted_suffixes(args), reader=readers.SCAN_READER)
    except ValueError as exc:
        raise BadUsageError(str(exc)) from exc
    for empty in empty_roots:
        # Not a usage error: the artifact is written, and the report HOLDS on the root.
        print(f"note: corpus root {empty.root} admitted no file at this scan's suffix scope.", file=sys.stderr)

    try:
        declared, dropped_facts = load_scan_facts(yaml.safe_load(args.facts.read_text(encoding='utf-8'))) if args.facts else ([], [])
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # The DOCUMENT's shape only. One bad fact inside a good document is dropped and reported.
        raise BadUsageError(f'--facts {args.facts}: {exc}') from exc
    for dropped in dropped_facts:
        print(f'note: fact {dropped.fact_id!r} was dropped unread: {dropped.reason}', file=sys.stderr)
    if not declared and dropped_facts:
        # Supplied, and every one was malformed — not the same fact as "none were supplied".
        print(f'note: {len(dropped_facts)} fact(s) supplied, all dropped, so the fact-verification check decides nothing.', file=sys.stderr)
    elif not declared:
        # NOT an error: a scan with no typed facts still benefits from the allowlist and the
        # hash. But it must be stated, or an all-clear reads as "the facts checked out".
        print('note: no typed facts were supplied, so the fact-verification check decides nothing.', file=sys.stderr)

    try:
        indicators = load_service_indicators(yaml.safe_load(args.service_indicators.read_text(encoding='utf-8'))) if args.service_indicators else None
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise BadUsageError(f'--service-indicators {args.service_indicators}: {exc}') from exc

    source_roots = tuple(args.source_root or ())
    problems: list[DeclarationProblem] = []
    # The render / action surface. Looked for under the SAME roots the citations resolve
    # against, so no flag is added and the parity table stays whole. A malformed declaration is
    # DROPPED with a report item naming it, never read as a surface: undeclared means "nothing to
    # compare", which never false-reds.
    try:
        render_surface = discover_render_surface(source_roots)
    except RenderActionsConfigError as exc:
        render_surface = None
        problems.append(DeclarationProblem(check=RENDER_ACTIONS_KEY, detail=str(exc)))
    if render_surface is not None:
        listed = ', '.join(render_surface.verbs) or 'none, by declaration'
        print(
            f'note: render surface: {len(render_surface.verbs)} verb(s) declared in {render_surface.source} ({listed}); '
            f'carried on the artifact as `{RENDER_ACTIONS_KEY}`.',
            file=sys.stderr,
        )
    elif source_roots:
        print(
            f'note: no `{RENDER_ACTIONS_KEY}:` declaration in a plurai.config.yaml under the source root(s), so the artifact carries '
            "no render surface and the backend's RENDER_ACTIONS check has nothing to compare an invented verb against.",
            file=sys.stderr,
        )

    # …and the closed-class detector overrides, read from the SAME configs under the
    # SAME roots. Both shells discover and pass them identically because the discovery is the
    # VENDORED one — a second reader here would let a customer's declaration mean one thing on
    # their machine and another on the backend, which is the drift this whole package exists to
    # remove. Declarations go IN through `run_scan_contract`; verdicts come out on the artifact. One
    # malformed ENTRY is dropped on its own, and named by a report item.
    closed_class_overrides, override_problems = discover_closed_class_overrides(source_roots)
    problems.extend(DeclarationProblem(check=CLOSED_CLASS_CONFIG_KEY, detail=detail) for detail in override_problems)
    for problem in problems:
        print(f'note: the `{problem.check}:` declaration was dropped: {problem.detail}', file=sys.stderr)
    if closed_class_overrides:
        print(
            f'note: {len(closed_class_overrides)} closed-class detector override(s) declared '
            f'({", ".join(sorted(closed_class_overrides))}). Each is matched to a `closed_set` fact BY SUBJECT and then '
            'VALIDATED by the same six gates, never trusted — one matching no fact is reported as a finding, not applied.',
            file=sys.stderr,
        )

    try:
        result = run_scan_contract(
            units=units,
            inventory=inventory,
            facts=declared,
            recorded_inventory=read_recorded_inventory(args.against) if args.against else None,
            probe_budget=args.probe_budget,
            corpus_id=args.corpus_id,
            load_warnings=warnings,
            probe_consent=args.probe_consent,
            source_roots=source_roots,
            state_schema_yaml=args.state_schema.read_text(encoding='utf-8') if args.state_schema else None,
            stateless_finding=args.stateless_finding,
            variables_yaml=args.variables.read_text(encoding='utf-8') if args.variables else None,
            service_indicators=indicators,
            closed_class_overrides=closed_class_overrides,
            overrules=overrules,
            dropped_facts=dropped_facts,
            empty_roots=empty_roots,
            declaration_problems=problems,
        )
    except UnknownGateError as exc:
        # A `scan.*` overrule naming no scan gate — a typo. Usage: the overrule file says something
        # this checker cannot honour, and judging it as "lifted nothing" would hide that.
        raise BadUsageError(f'--overrule {args.overrule}: {exc}') from exc

    _run_smoke_probes(result, agent_command=args.agent_command, budget=args.smoke_budget, timeout=args.probe_timeout)

    rendered = result.report.render_markdown()
    payload = artifact_payload(result)
    if render_surface is not None:
        # An SDK-OWNED field on the backend-shaped payload: ``artifact_payload`` is vendored and
        # emits nothing for a surface its shell never reads. Top level, beside ``corpus_id`` and
        # ``source_inventory`` (the other carried inputs); ABSENT when nothing declared — an empty
        # ``verbs`` list here is a declaration, and only ever that.
        payload[RENDER_ACTIONS_KEY] = render_surface.as_record()
    _atomic_write_text(args.out / ARTIFACT_FILENAME, json.dumps(payload, indent=2, ensure_ascii=False))
    _atomic_write_text(args.out / REPORT_FILENAME, rendered)
    # INTERNAL: never inside the artifact and never beside it — the overrule reasons it carries must
    # not reach the uploaded, committed scan_contract.json or the directory it is committed from.
    record = gate_record_path(args.out)
    _prepare_gate_record_dir(record.parent)
    _atomic_write_text(record, json.dumps(gate_authority_payload(result), indent=2, ensure_ascii=False))
    print(rendered)
    print(f'artifact: {args.out / ARTIFACT_FILENAME}   report: {args.out / REPORT_FILENAME}')
    # One count line, like the ingest and runner commands: counts only, the reasons stay in the record.
    if (gate_line := result.report.gate_count_line(record_path=str(record))) is not None:
        print(gate_line, file=sys.stderr)

    # ONE gate: `report.proceed`, which the vendored checker computed AFTER applying the overrules. A
    # held finding exits ``SCAN_HELD``; nothing else the report says moves the exit code.
    if not result.report.proceed:
        print(result.blocking_message(), file=sys.stderr)
        return exit_codes.SCAN_HELD
    return exit_codes.OK


def _run_smoke_probes(result: ScanContractResult, *, agent_command: str | None, budget: int, timeout: float) -> None:
    """Fire the derived smoke probes at the live agent and fold the transcripts into the report.

    The set is rebuilt from the report's own section rather than re-derived: the section IS the
    artifact this command publishes, and deriving a second copy here would be a second producer
    of the probes a customer reads.

    Without ``--agent-command`` nothing is fired and the report is left as the derivation wrote
    it — which already reads DERIVED ONLY, naming the probes it did not run. The console note is
    for the operator standing here, not a correction to the artifact.

    Consent is enforced one layer down, on the set itself, so the gate is the same whether probes
    are fired from here or from a later call over the persisted artifact.
    """
    section = result.report.sections.get('scan_time_smoke_probes')
    if not isinstance(section, dict):
        return
    smoke = SmokeProbeSet.from_dict(section)
    if agent_command is None:
        if smoke.probes:
            print(f'note: {len(smoke.probes)} smoke probe(s) were derived and NOT fired — pass --agent-command to run them.', file=sys.stderr)
        return
    try:
        transport = subprocess_transport(agent_command, timeout=timeout)
    except ValueError as exc:
        # A usage error, not a crash: the traceback tells the operator nothing the message does not.
        raise BadUsageError(f'--agent-command: {exc}') from exc
    execute_smoke_probes(smoke, transport, budget=budget)
    record_execution_in_report(result.report, smoke)


def _cmd_recheck(args: argparse.Namespace) -> int:
    """Re-hash the sources, report drift, and — with ``--facts`` — RE-VERIFY the typed facts.

    0 for every VERDICT. Drift is advisory, and a command that can exit non-zero on a verdict is a
    command that can block the run it was only ever meant to annotate. A refuted fact is reported
    just as loudly and is still not a blocking exit: what to do about it is the suite owner's
    call. The one non-zero exit is ``INTERNAL_FAULT``, raised from :func:`main` — a
    defect in this checker is not a drift verdict either.
    """
    outcome = recheck_source_drift(args.corpus, args.against, suffixes=_admitted_suffixes(args))
    print(outcome.summary())
    if args.facts is None:
        return 0

    # The text is needed now, so the hash-only shortcut above does not apply: load_corpus_roots
    # additionally converts, YAML-parses and flattens every source (~170x the cost of hashing),
    # which is exactly what fact verification reads. At the SAME scope the drift walk used — the
    # baseline's — so the facts are re-verified over the files the recording check decided them over.
    scope = outcome.scope or _admitted_suffixes(args)
    try:
        units, _inventory, _warnings, empty_roots = load_corpus_roots(args.corpus, suffixes=scope, reader=readers.SCAN_READER)
    except ValueError as exc:
        raise BadUsageError(str(exc)) from exc
    if empty_roots:
        # A usage error HERE, unlike `check`: recheck writes no artifact a finding could ride.
        raise BadUsageError(
            f'corpus root(s) {[str(e.root) for e in empty_roots]} hold no file this scan reads (it reads '
            f'{", ".join(scope)}) — re-verifying facts over them proves nothing.'
        )
    try:
        declared, dropped_facts = load_scan_facts(yaml.safe_load(args.facts.read_text(encoding='utf-8')))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise BadUsageError(f'--facts {args.facts}: {exc}') from exc
    for dropped in dropped_facts:
        print(f'note: fact {dropped.fact_id!r} was dropped unread: {dropped.reason}', file=sys.stderr)
    verification = verify_facts(declared, units)
    refuted = [v.fact.fact_id for v in verification.contradicted]
    if not refuted:
        print(f'{len(declared)} declared fact(s) re-verified; none is refuted by the sources as they are now.')
        print(verification.coverage_note())
        return 0
    print(f'{len(refuted)} fact(s) are now FALSE against the sources: {", ".join(refuted)}')
    dependents = fact_dependents(
        refuted,
        policies=yaml.safe_load(args.policies.read_text(encoding='utf-8')) if args.policies else None,
        dataset_metadata=yaml.safe_load(args.dataset_metadata.read_text(encoding='utf-8')) if args.dataset_metadata else None,
    )
    if args.policies is None and args.dataset_metadata is None:
        # Naming the refuted fact without naming what stands on it is half the drift story. Said
        # out loud, because "no dependents were listed" otherwise reads as "nothing depends on it".
        print(
            'No suite was named, so the dependent scenarios/evals were NOT resolved. Re-run with --policies '
            'and --dataset-metadata to see exactly what must be rebuilt.',
            file=sys.stderr,
        )
        return 0
    print(dependents.render())
    return 0


def _expanded_path(raw: str) -> Path:
    return Path(expand_user_path(raw))


def _existing_dir(raw: str) -> Path:
    path = Path(expand_user_path(raw))
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'{raw}: not a directory')
    return path


def _existing_path(raw: str) -> Path:
    path = Path(expand_user_path(raw))
    if not path.exists():
        raise argparse.ArgumentTypeError(f'{raw}: does not exist')
    return path


def _existing_file(raw: str) -> Path:
    path = Path(expand_user_path(raw))
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'{raw}: not a file')
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = exit_codes.UsageParser(
        prog='plurai-scan-contract',
        description="Execute a scan's conclusions against the corpus they were drawn from, before generation spends.",
    )
    sub = parser.add_subparsers(dest='command')

    check = sub.add_parser('check', help="Verify the scan's conclusions and write the pre-generation findings report.")
    check.add_argument('--corpus', action='append', type=_existing_path, required=True, help='Corpus root to scan: a directory, or one file. Repeatable.')
    check.add_argument('--facts', type=_existing_file, default=None, help='Typed scan-facts YAML/JSON to verify against the corpus.')
    check.add_argument('--out', type=_expanded_path, default=Path(), help='Directory the artifact + report are written to.')
    check.add_argument('--corpus-id', default=None, help='The built corpus id the source fingerprint is stored beside.')
    check.add_argument('--against', type=_existing_file, default=None, help='A previous scan_contract.json, to report source drift.')
    check.add_argument('--probe-budget', type=int, default=DEFAULT_PROBE_BUDGET, help='How many verified absences generate an adversarial probe.')
    add_overrule_argument(check)
    check.add_argument(
        '--probe-consent',
        action='store_true',
        help='The customer consented to scan-time probes of the LIVE agent. Probes are DERIVED and reported either way; '
        'without this flag NONE is executed, and each is recorded not-run with that reason.',
    )
    check.add_argument(
        '--agent-command',
        default=None,
        help="Command that takes one probe ask on stdin and prints the agent's reply on stdout. Required (with "
        '--probe-consent) to actually fire the derived smoke probes at the live agent.',
    )
    check.add_argument('--smoke-budget', type=int, default=DEFAULT_SMOKE_BUDGET, help='Hard cap on how many smoke probes are fired at the live agent.')
    check.add_argument('--probe-timeout', type=float, default=DEFAULT_PROBE_TIMEOUT, help='Seconds one probe may take before its transport is abandoned.')
    # The plugin's `scripts/scan_contract.sh` runs whichever `plurai-scan-contract` is on the
    # client's venv — this one — with the flags below, so a flag missing here is an argument-parse
    # failure at the call site. The vendored package under `plurai/scan_contract/` accepts every
    # parameter below; only the wiring lives here.
    check.add_argument(
        '--source-root',
        action='append',
        type=_existing_dir,
        default=None,
        help="Tree each fact's `source:` citation is resolved against (the agent module). Repeatable.",
    )
    check.add_argument('--state-schema', type=_existing_file, default=None, help='scenario_state_schema.yaml, to decide whether `agent_context` is governed.')
    check.add_argument(
        '--stateless-finding',
        default='',
        help='What the scan established about this agent reading no turn-0 state. Stands in for --state-schema, never beside it.',
    )
    check.add_argument('--variables', type=_existing_file, default=None, help='scenario_variables.yaml, for the typed `typical_complications` pool.')
    check.add_argument(
        '--prd',
        type=_existing_file,
        default=None,
        help='simulation_prd.md. Accepted and not read: the complication-pool check runs at generate-scenarios job start.',
    )
    check.add_argument(
        '--suffixes',
        action='append',
        default=None,
        help='Extra suffix of PLAIN-TEXT files to admit into the corpus (repeatable; "ts" or ".ts"). ADDS to the default set — a '
        f'scan that reads FEWER files than it thinks is the failure this flag exists for. A binary is refused, never searched. '
        f'Default: every format the SDK reads — {readers.describe()}.',
    )
    check.add_argument(
        '--service-indicators',
        type=_existing_file,
        default=None,
        help='YAML mapping `service name -> [ENV_VAR, ...]`, MERGED into the built-in table so a vendor it does not list can '
        'still contradict a "no external services" claim.',
    )
    check.set_defaults(func=_cmd_check)

    recheck = sub.add_parser('recheck', help='Re-hash the sources and report drift against a recorded scan. Always exits 0.')
    recheck.add_argument(
        '--corpus', action='append', type=_existing_path, required=True, help='Corpus root to re-hash. Repeatable, in the same order as the recording run.'
    )
    recheck.add_argument('--against', type=_existing_file, required=True, help='The scan_contract.json whose source_inventory is the baseline.')
    recheck.add_argument('--facts', type=_existing_file, default=None, help='Typed scan-facts YAML/JSON to RE-VERIFY against the sources as they are now.')
    recheck.add_argument('--policies', type=_existing_file, default=None, help="A suite's policies_config.yaml, to name the evals a refuted fact takes down.")
    recheck.add_argument(
        '--dataset-metadata',
        type=_existing_file,
        default=None,
        help="A scenario set's dataset_metadata.yaml, to name the scenarios a refuted fact takes down.",
    )
    recheck.add_argument(
        '--suffixes',
        action='append',
        default=None,
        help='Extra file suffix to admit, repeatable. Used ONLY for a baseline that recorded no scope: a baseline that recorded its '
        'scope (source_inventory.suffixes) is re-walked at exactly that scope, whatever is passed here.',
    )
    recheck.set_defaults(func=_cmd_recheck)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Exit ``OK`` / ``SCAN_HELD`` / ``USAGE`` / ``REFUSED`` / ``INTERNAL_FAULT`` (:mod:`plurai.exit_codes`).

    ``OK`` no held finding. ``SCAN_HELD`` a HELD finding (an un-overruled holding one). ``USAGE`` a usage error or a
    malformed input document. ``INTERNAL_FAULT`` THE CHECKER ITSELF FAILED.

    The last one keeps a crash apart from a finding: an unexpected exception would otherwise reach
    the interpreter's own handler and exit 1 — the code of a failed policy — so
    the plugin's ``scripts/scan_contract.sh`` would read a bug in the checker as "the scan asserted something
    false". The traceback is printed rather than swallowed — the code says WHO failed, the
    traceback says where.
    """
    utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, 'func', None) is None:
        parser.print_help(sys.stderr)
        return exit_codes.USAGE
    try:
        return args.func(args)
    except BadUsageError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return exit_codes.USAGE
    except Exception as exc:  # noqa: BLE001 - a defect in this checker is not a finding against the scan
        print(f'error: the checker itself failed during `{args.command}` ({type(exc).__name__}: {exc}).', file=sys.stderr)
        print(
            f'error: exit {exit_codes.INTERNAL_FAULT} means NOTHING was verified — it is not a finding against the scan. '
            'The traceback below is a bug in plurai-scan-contract.',
            file=sys.stderr,
        )
        traceback.print_exc()
        return exit_codes.INTERNAL_FAULT


if __name__ == '__main__':
    sys.exit(main())
