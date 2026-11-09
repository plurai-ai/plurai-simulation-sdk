"""One command invocation's gate findings: the collector, the exit funnel, and the internal record.

ONE COLLECTOR. :func:`run_invocation` opens a :class:`GateAuthority` for the command and holds it in
a ContextVar for the whole invocation; every gate site reaches it through :func:`current` and
reports with :meth:`GateAuthority.emit` (a finding) or :meth:`GateAuthority.ran` (the gate ran and
found nothing). Tasks and per-trial contexts inherit the same object, so a ``--trials N`` run
collects every trial's findings in one place and an overrule is judged against all of them.

ONE EXIT FUNNEL. :func:`held_exit` is the only place a finding changes an exit code: a hard
finding, or a holding advisory finding no ``--overrule`` lifted, returns the command's existing
refusal code and names each held finding's gate id, fingerprint and remedy on stderr. A finding
that holds nothing never reaches an exit code, and nothing that writes a command's own result
(``run_summary.json``, a manifest) reads this module, except the trials aggregate, which counts a
cell only when an overrule lifted its hold.

ONE SETTLE POINT. When the invocation ends — on every path, an exception included — the record is
written to the :class:`RecordTarget` the command claimed with :meth:`GateAuthority.claim`, replacing the
previous invocation's (a target that records only when there is something to record leaves no file
for an invocation that found nothing and carried no overrule). It is INTERNAL: overrule reasons and gate evidence can describe the customer's
material, so it is never merged into another artifact, never uploaded and never rendered for the
customer. The terminal gets one count line and no reasons.

WHERE THE FUNNEL IS CALLED. ``plurai-run-simulation`` calls it after each pre-spend rung that can
report a hard finding, and once at the funnel. A gate that runs anywhere else — a preflight inside a run strategy, an ingest subcommand's
body — calls ``held_exit(refusal_code=<that command's refusal code>)`` right after its own emit and
returns the code through the command's existing early-exit path. There is no second funnel: every
stop goes through :func:`held_exit`. The one command outside it is ``plurai-scan-contract check``:
its gates are shared with the backend's shell, so the VENDORED checker applies its overrules and its
``report.proceed`` is the exit (1 while held) — identically in both shells.

OUTSIDE AN INVOCATION. Code that runs with no open invocation — a ``plurai.child`` process, a test that
drives a run strategy below ``main()``, a thread that did not copy the context — reaches an
UNATTENDED collector from :func:`current`: it validates the gate id, logs each emitted finding at
WARNING (gate id, authority, subject, fingerprint) and keeps nothing, so :func:`held_exit` there
returns ``None`` and no record is written. A finding has an effect only inside the invocation of the
command that owns the funnel; a gate that must stop a child acts through the child's own failure path.

Routing: an overrule naming an SDK gate (:func:`~plurai.gates.catalogue.is_sdk_gate`) is applied
here with the vendored :func:`~plurai.gates.apply.apply_overrules`. Any other id belongs to the
backend: the runner forwards it on every ``/finalize`` and ``/rescore`` it submits
(:meth:`GateAuthority.forwarded_overrules`) and merges each answer the backend returns
(:meth:`GateAuthority.answer_backend`). An id no backend answer acknowledged — a command with no
such request, a backend that predates the field (it ignores the key) or one that refused it
(:meth:`GateAuthority.backend_declined`) — is recorded ``unacknowledged``: never applied, never stale.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from plurai.gates.apply import apply_overrules
from plurai.gates.catalogue import CELL_HOLDS, gate_spec, is_sdk_gate
from plurai.gates.cli import OverrulesFile
from plurai.gates.contract import HARD_AUTHORITIES, GateFinding, Overrule, OverruleOutcome, OverruleOutcomeKind
from plurai.identity import sdk_identity

#: The runner's record, beside ``run_summary.json`` in ``--out`` (never inside it).
GATE_AUTHORITY_FILENAME: Final[str] = 'gate_authority.json'

GATE_AUTHORITY_SCHEMA: Final[str] = 'gate-authority-v1'

#: The ``command`` of the collector :func:`current` hands out when no invocation is open.
UNATTENDED_COMMAND: Final[str] = '<no invocation>'

logger = logging.getLogger(__name__)

#: How one backend answer outranks another when several requests answered the same overrule (each scenario's
#: ``/finalize`` answers every forwarded overrule): the union of the backend's findings and evaluated gates read as one
#: invocation, the same way the SDK's own overrules are judged once against every trial's findings.
_BACKEND_OUTCOME_RANK: Final[Mapping[OverruleOutcomeKind, int]] = {'not_evaluated': 0, 'stale': 1, 'not_holding': 2, 'applied': 3, 'refused': 4}


def _key(overrule: Overrule) -> tuple[str, str]:
    return (overrule.gate_id, overrule.fingerprint)


@dataclass(frozen=True)
class RecordTarget:
    """Where an invocation's record goes, and when it is written."""

    path: Path
    #: Write a record even when the invocation emitted no finding and carried no overrule. The runner does,
    #: so a reader of ``--out`` always finds the record of the invocation that wrote it.
    when_empty: bool
    #: Called with the record's directory before it is written (e.g. to make a private directory
    #: self-ignoring); ``None`` when the directory needs nothing.
    prepare: Callable[[Path], None] | None


@dataclass
class GateAuthority:
    """The gate findings of one command invocation and the overrules it was given."""

    command: str
    overrules_file: OverrulesFile | None
    findings: list[GateFinding] = field(default_factory=list)
    #: Every SDK gate that RAN in this invocation, finding or not: what separates ``stale`` from ``not_evaluated``.
    evaluated: set[str] = field(default_factory=set)
    target: RecordTarget | None = None
    #: The backend's answer to each forwarded overrule, keyed by ``(gate_id, fingerprint)``, merged across requests.
    backend_outcomes: dict[tuple[str, str], OverruleOutcome] = field(default_factory=dict)
    #: Why the backend refused the ``overrules`` field, once it has: nothing is forwarded after that. ``None`` otherwise.
    backend_declined_reason: str | None = None
    #: Each backend ``gate_authority`` block (or outcome row) that did not validate, RAW: its outcomes could not be
    #: recorded, so it is kept verbatim in the internal record rather than lost.
    backend_unparsed: list[Any] = field(default_factory=list)
    #: Overrule markers stripped off a backend cell that no recorded ``applied`` outcome accounts for. Kept here — never
    #: on the cell, which reaches customer artifacts — so the restored verdict stays traceable.
    backend_unaccounted_markers: list[dict[str, Any]] = field(default_factory=list)
    #: The fingerprints :func:`held_exit` NAMED on stderr (every call of it in this invocation). ``held`` in the record is
    #: these plus every cell hold (:data:`~plurai.gates.catalogue.CELL_HOLDS`); a hard or holding finding that holds the
    #: COMMAND and was emitted after the funnel (or never put through it) is ``held_after_exit``.
    named_at_exit: set[str] = field(default_factory=set)

    @property
    def overrules(self) -> tuple[Overrule, ...]:
        return self.overrules_file.overrules if self.overrules_file is not None else ()

    @property
    def local_overrules(self) -> tuple[Overrule, ...]:
        """Overrules naming an SDK gate: applied here."""
        return tuple(o for o in self.overrules if is_sdk_gate(o.gate_id))

    @property
    def backend_overrules(self) -> tuple[Overrule, ...]:
        """Overrules naming a backend gate: forwarded, and answered by the backend."""
        return tuple(o for o in self.overrules if not is_sdk_gate(o.gate_id))

    @property
    def unacknowledged(self) -> tuple[Overrule, ...]:
        """Backend overrules no backend answer acknowledged (none carried them, or the backend did not answer)."""
        return tuple(o for o in self.backend_overrules if _key(o) not in self.backend_outcomes)

    def forwarded_overrules(self) -> tuple[Overrule, ...]:
        """What a backend request of this invocation carries: every backend overrule, until the backend refused the field."""
        return () if self.backend_declined_reason is not None else self.backend_overrules

    def backend_declined(self, endpoint: str, detail: str) -> None:
        """The backend refused a request for the ``overrules`` it carried (a build that applies none, or an overrule it
        could not validate). Nothing is forwarded from here on, so the caller re-sends without them, and every backend
        overrule stays ``unacknowledged`` unless an earlier answer already acknowledged it."""
        if self.backend_declined_reason is None:
            self.backend_declined_reason = f'{endpoint}: {detail}'
            # The endpoint only: a validation answer echoes the body it refused, reasons included, and those stay in the record.
            logger.warning('%s refused the forwarded gate overrules; they stay unacknowledged (the answer is in the gate record)', endpoint)

    def answer_backend(self, block: Mapping[str, Any] | None) -> None:
        """Merge one backend result's ``gate_authority`` block. ``None`` (a backend that predates the field, or a request
        that carried no overrule) acknowledges nothing; an outcome for an overrule this invocation did not forward is
        ignored. Outcomes are validated ROW BY ROW, so one bad row never voids the rest; a block whose shape cannot be read,
        and each row that does not validate, is kept raw in ``backend_unparsed`` and logged."""
        if block is None:
            return
        rows = block.get('overrules') if isinstance(block, Mapping) else None
        if not isinstance(rows, list):
            self.backend_unparsed.append(block)
            logger.warning('a backend gate_authority block has no readable `overrules` list; it is kept unparsed in the gate record')
            return
        outcomes: list[OverruleOutcome] = []
        for row in rows:
            try:
                outcomes.append(OverruleOutcome.model_validate(row))
            except ValidationError as exc:
                # The error count only: a validation message echoes its input, and the row carries an overrule reason.
                logger.warning('a backend overrule outcome did not validate (%d error(s)); it is kept unparsed in the gate record', exc.error_count())
                self.backend_unparsed.append(row)
        sent = {_key(o) for o in self.backend_overrules}
        for outcome in outcomes:
            key = _key(outcome.overrule)
            if key not in sent:
                continue
            held = self.backend_outcomes.get(key)
            if held is None or _BACKEND_OUTCOME_RANK[outcome.outcome] > _BACKEND_OUTCOME_RANK[held.outcome]:
                self.backend_outcomes[key] = outcome

    def applied_backend_fingerprints(self) -> frozenset[str]:
        """The fingerprints a recorded backend outcome says an overrule LIFTED — what accounts for a cell's marker."""
        return frozenset(fp for (_gate, fp), outcome in self.backend_outcomes.items() if outcome.outcome == 'applied')

    def claim(self, target: RecordTarget) -> None:
        """From here on this invocation's record goes to ``target``. The previous invocation's record there is
        removed now, so a process that dies before the settle point leaves no stale record behind."""
        self.target = target
        try:
            target.path.unlink(missing_ok=True)
        except OSError as exc:
            print(f'WARNING: could not remove the previous gate record {target.path}: {exc}', file=sys.stderr)

    def ran(self, gate_id: str) -> None:
        """The gate ran in this invocation. Its id must be declared (an undeclared id raises)."""
        gate_spec(gate_id)
        self.evaluated.add(gate_id)

    def emit(self, finding: GateFinding) -> None:
        """Record ``finding``. Whether it stops the command is decided at :func:`held_exit`, never here."""
        self.ran(finding.gate_id)
        self.findings.append(finding)

    def _lifted(self, finding: GateFinding) -> bool:
        return any((o.gate_id, o.fingerprint) == (finding.gate_id, finding.fingerprint) for o in self.local_overrules)

    def held(self) -> list[GateFinding]:
        """Every finding that stops the command: each hard one, and each holding advisory one no overrule lifted."""
        return [f for f in self.findings if f.authority in HARD_AUTHORITIES or (f.holds and not self._lifted(f))]

    def _answered(self, finding: GateFinding) -> bool:
        """Whether ``finding``'s held effect took place: the funnel named it (the exit code answered for it), or it
        holds a trial cell (:data:`~plurai.gates.catalogue.CELL_HOLDS`), which the trials aggregation holds whenever
        the finding is emitted."""
        return finding.fingerprint in self.named_at_exit or finding.gate_id in CELL_HOLDS

    def held_at_exit(self) -> list[GateFinding]:
        """The held findings whose effect took place: each one :func:`held_exit` named, and each cell hold."""
        return [f for f in self.held() if self._answered(f)]

    def held_after_exit(self) -> list[GateFinding]:
        """Held findings that hold the COMMAND and that the funnel never named: emitted after it ran (or on a path
        that never ran it), so the exit code does not reflect them. Recorded apart so a reader never takes them for
        what stopped the command."""
        return [f for f in self.held() if not self._answered(f)]

    def outcomes(self) -> list[OverruleOutcome]:
        """What each overrule did: an SDK-gate one judged once against every finding of the invocation, then each
        backend one the backend answered (its strongest answer across requests)."""
        local = apply_overrules(self.findings, self.local_overrules, evaluated_gates=self.evaluated)
        return [*local, *(self.backend_outcomes[k] for o in self.backend_overrules if (k := _key(o)) in self.backend_outcomes)]

    def record(self, *, exit_code: int | None) -> dict[str, Any]:
        return {
            'schema': GATE_AUTHORITY_SCHEMA,
            'written_at': datetime.now(UTC).isoformat(),
            'command': self.command,
            'sdk_identity': sdk_identity(),
            'exit_code': exit_code,
            'overrules_file': self.overrules_file.as_record() if self.overrules_file is not None else None,
            'findings': [f.model_dump(mode='json') for f in self.findings],
            'held': [f.fingerprint for f in self.held_at_exit()],
            'held_after_exit': [f.fingerprint for f in self.held_after_exit()],
            'overrules': [o.model_dump(mode='json') for o in self.outcomes()],
            'unacknowledged': [o.model_dump(mode='json') for o in self.unacknowledged],
            'backend_declined': self.backend_declined_reason,
            'backend_unparsed': list(self.backend_unparsed),
            'backend_unaccounted_markers': list(self.backend_unaccounted_markers),
        }

    def count_line(self) -> str | None:
        """The one terminal line, or ``None`` when there is nothing to count. Counts only — no reasons."""
        if not self.findings and not self.overrules:
            return None
        kinds = [o.outcome for o in self.outcomes()]
        where = f'; see {self.target.path}' if self.target is not None else ''
        return (
            f'[gate] {len(self.findings)} finding(s), {len(self.held_at_exit())} held, {len(self.held_after_exit())} held after exit, '
            f'{kinds.count("applied")} overruled, '
            f'{kinds.count("stale")} stale, {len(self.unacknowledged)} unacknowledged{where}'
        )

    def settle(self, *, exit_code: int | None) -> None:
        """Write the record (when a target was claimed and there is something to record, or the target records
        when empty) and print the count line. Best-effort: a write failure is reported on stderr with every held
        fingerprint, which is what a caller needs to overrule."""
        target = self.target
        if target is not None and (target.when_empty or self.findings or self.overrules):
            try:
                if target.prepare is not None:
                    target.prepare(target.path.parent)
                _write_record(target.path, self.record(exit_code=exit_code))
            except (OSError, TypeError, ValueError) as exc:
                held = ', '.join(f'{f.gate_id} {f.fingerprint}' for f in self.held()) or 'none'
                print(f'WARNING: the gate record {target.path} could not be written ({exc}); held: {held}', file=sys.stderr)
        if (line := self.count_line()) is not None:
            print(line, file=sys.stderr)


class _Unattended(GateAuthority):
    """The collector :func:`current` hands out when no invocation is open: validates, logs, keeps nothing."""

    def claim(self, target: RecordTarget) -> None:
        logger.warning('gate record %s not claimed: no gate-authority invocation is open', target.path)

    def emit(self, finding: GateFinding) -> None:
        gate_spec(finding.gate_id)
        logger.warning(
            'gate finding %s (%s) on %r emitted with no open invocation; not recorded and not held [fingerprint %s]',
            finding.gate_id,
            finding.authority,
            finding.subject,
            finding.fingerprint,
        )


def _write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


_CURRENT: ContextVar[GateAuthority] = ContextVar('plurai_gate_authority')


def current() -> GateAuthority:
    """The open invocation's collector, or an UNATTENDED one when none is open (see the module docstring)."""
    authority = _CURRENT.get(None)
    return authority if authority is not None else _Unattended(command=UNATTENDED_COMMAND, overrules_file=None)


def run_invocation(*, command: str, overrules_file: OverrulesFile | None, record: RecordTarget | None, body: Callable[[], int]) -> int:
    """Run ``body`` as one invocation of ``command`` and return its exit code, settling the record on every
    path — an exception included (recorded with ``exit_code: null`` and re-raised). ``record=None`` leaves
    the claim to the command (:meth:`GateAuthority.claim`), for one that must not write before it owns its
    output directory. A command that maps its own errors to an exit code does so INSIDE ``body``, so the
    record carries the code the process exits with."""
    authority = GateAuthority(command=command, overrules_file=overrules_file)
    if record is not None:
        authority.claim(record)
    token = _CURRENT.set(authority)
    exit_code: int | None = None
    try:
        exit_code = code = body()
    finally:
        _CURRENT.reset(token)
        authority.settle(exit_code=exit_code)
    return code


def held_exit(*, refusal_code: int) -> int | None:
    """THE exit funnel. ``refusal_code`` when a finding holds the command, after naming each held finding on
    stderr; ``None`` otherwise. ``refusal_code`` is the command's existing refusal code, passed by the caller."""
    authority = current()
    held = authority.held()
    if not held:
        return None
    # The snapshot the record's `held` reads: what THIS funnel named. A finding held later is `held_after_exit`.
    authority.named_at_exit.update(f.fingerprint for f in held)
    for finding in held:
        print(
            f'ERROR: held by gate {finding.gate_id} ({finding.authority}) on {finding.subject!r}: {finding.remedy} [fingerprint {finding.fingerprint}]',
            file=sys.stderr,
        )
    return refusal_code
