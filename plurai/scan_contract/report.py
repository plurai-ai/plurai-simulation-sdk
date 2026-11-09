"""The ONE pre-generation findings report — emitted after scan, before any generation spends.

Everything upstream of this module produces a typed artifact that a job would otherwise carry
quietly to its own workdir. Findings left there stay buried in a job artifact nobody opens until
a run comes back wrong, after the customer has already paid for generation. This report is the
single place those artifacts surface, with one signal a human can act on: PROCEED, or FIX FIRST.

It is PURE AGGREGATION. Nothing here re-derives, re-decides or re-words a verdict reached
upstream — a report that computes its own version of a fact is a second implementation, and the
two drift. Every item carries the provenance of the check that produced it.

ONE SIGNAL, three gates. Only three items HOLD generation, each minted as a holding advisory
:class:`~plurai.gates.contract.GateFinding` (:mod:`plurai.scan_contract.gate_ids`): a fact the
corpus refutes, a closed set whose author required a detector that none passed, and a corpus root
that admitted no file. Everything else is a plain report item — its evidence already rides the
artifact. A caller lifts a hold with an overrule bound to the finding's fingerprint
(:meth:`PreGenerationReport.apply_overrules`); the verdict underneath never changes, and the
overrule never renders here.

Three rendering rules:

* **Fix-first items render at the TOP.** A contradicted fact under a heading three sections down
  is a finding nobody reads.
* **An empty report renders as an explicit all-clear.** Silence is the shape a check that never
  ran also has, and the two must never look alike.
* **A single finding's prose SAMPLES a long list and names where the whole one lives.** An
  enumeration inlined into one item is a markdown line no human reads and a second copy of an
  artifact that already exists; a sample plus a pointer is the same information at a size a
  reader can act on. This governs a list rendered INSIDE one item's detail — items that are
  one-per-entry by design (the per-template advisories) each carry their own provenance and are
  bounded by the entry count, not by a join.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from plurai.gates.apply import applied_fingerprints, apply_overrules, make_finding
from plurai.gates.contract import GateAuthorityBlock, GateFinding, Overrule, OverruleOutcome
from plurai.scan_contract._suite_findings import CheckLedger
from plurai.scan_contract.closed_class_gate import GATE_REMEDY, ClosedClassDecision
from plurai.scan_contract.citations import CITATION_NOT_A_PATH, CITATION_RESOLVED, CITATION_UNCHECKED, CITATION_UNSUPPORTED, CitationAudit
from plurai.scan_contract.corpus import DriftReport, EmptyRoot, SourceInventory
from plurai.scan_contract.facts import (
    SEVERITY_ORDER,
    CONTRADICTED,
    UNVERIFIED_AT_SCAN,
    UNVERIFIED_NO_STATIC_ROUTE,
    UNVERIFIED_NOT_DERIVABLE,
    UNVERIFIED_UNRESOLVABLE,
    VERIFIED,
    DroppedFact,
    FactVerification,
)
from plurai.scan_contract.gate_ids import CLOSED_CLASS_DETECTOR_MISSING, CORPUS_ROOT_EMPTY, FACT_CONTRADICTED, SCAN_GATE_PREFIX
from plurai.scan_contract.inputs import STATE_UNGOVERNED, StateGovernance
from plurai.scan_contract.probes import ProbeSet, probe_budget_note, suppression_targets
from plurai.scan_contract.services import ExternalServiceAudit
from plurai.scan_contract.smoke_probes import SmokeProbeSet
from plurai.scan_contract.urls import UrlAllowlist

SCHEMA_VERSION = 'pre-generation-findings-v1'

#: Every check this report owes a ledger entry for. Closed, and seeded as ``blocked`` before the
#: sections run, so a section added under an ``if`` cannot simply vanish from ``checks`` and
#: render as though the check did not exist — the silence-vs-blocked collapse this module is
#: written against. The sibling suite report uses the same invariant for the same reason.
CHECK_NAMES: tuple[str, ...] = (
    'typed_fact_verification',
    'fact_source_citations',
    'placeholder_url_compilation',
    'corpus_source_hash',
    'corpus_source_drift',
    'absence_probes',
    'scan_time_smoke_probes',
    'agent_context_governance',
    'external_service_claims',
    'corpus_load',
    'closed_class_detectors',
)

SEVERITY_FIX_FIRST = 'fix_first'
SEVERITY_ADVISORY = 'advisory'
SEVERITY_INFO = 'info'

#: The artifact key carrying the COMPLETE licensed-URL set — the machine copy, and the only one
#: anything reads (the backend's scenario job reads exactly this path). ONE spelling, because
#: both the prose pointer and the section's own ``urls_published_at`` name it: two hand-written
#: copies of a key are two things to update when the key moves, and nothing compares them.
ALLOWLIST_URLS_KEY = 'url_allowlist.urls'

#: Where the complete set is published, named in the prose that samples it — a reader who needs
#: every entry must never have to guess whether one was published whole.
ALLOWLIST_POINTER = f'the machine artifact at `{ALLOWLIST_URLS_KEY}` in scan_contract.json'

#: How many entries a sampled detail names before it defers to the pointer. A catalog agent
#: licenses four figures of URLs, and an enumeration of those is a markdown line no human reads,
#: carrying nothing the machine artifact does not already hold.
SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class DeclarationProblem:
    """A ``plurai.config.yaml`` declaration the shell could not read, so the scan ran WITHOUT it.

    ``check`` is the declaration's own config key (``render_actions``, ``closed_class_detectors``).
    An undeclared surface or override means "nothing to compare", which never false-reds, so the
    scan proceeds and the report names what was dropped.
    """

    check: str
    detail: str


@dataclass(frozen=True)
class ReportItem:
    """One finding, with the check that produced it named."""

    severity: str
    check: str
    summary: str
    detail: str = ''
    #: The AUTHOR's declared severity for the underlying fact, when there is one. Distinct from
    #: ``severity`` above, which is this report's own act-now axis (fix_first / advisory / info):
    #: a contradicted `critical` fact and a contradicted `low` one are both fix-first, and
    #: without the declaration they render identically in a list a human triages.
    declared_severity: str | None = None
    #: Where the fact/URL/source came from — the unit id, the field path, the document.
    provenance: tuple[str, ...] = ()
    #: Set ONLY on an item that holds generation: the scan-gate finding it is (``gate_ids``), minted
    #: where the item is. Such an item is fix-first until a caller overrules the finding; nothing
    #: without one is.
    finding: GateFinding | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'severity': self.severity, 'check': self.check, 'summary': self.summary}
        if self.detail:
            out['detail'] = self.detail
        if self.declared_severity:
            out['declared_severity'] = self.declared_severity
        if self.provenance:
            out['provenance'] = list(self.provenance)
        return out


@dataclass
class PreGenerationReport:
    """What the scan established, and whether generation should start."""

    schema: str = SCHEMA_VERSION
    items: list[ReportItem] = field(default_factory=list)
    #: Section name -> already-typed payload from the check that owns it. Verbatim, not reworded.
    sections: dict[str, Any] = field(default_factory=dict)
    #: Fact ids the scan carried forward WITHOUT deciding — declared with no executable check, so
    #: there was no static route to test them. A DIFFERENT fact about the report from
    #: ``unexecuted_checks``: the check ran and reported honestly, over premises it could not
    #: test. Generation builds on them either way, which is why the headline names the count —
    #: a PROCEED over asserted premises is provisional, and the loop has to know which are soft.
    provisional_facts: list[str] = field(default_factory=list)
    #: What each check DID — the house's three-valued ledger, not a second implementation of it.
    #: ``ran`` / ``flagged`` / ``blocked`` is exactly the distinction this report needs (a check
    #: that could not run must not render like one that ran and found nothing), and encoding it
    #: as English prose made a consumer substring-match to find the blocked ones.
    checks: CheckLedger = field(default_factory=CheckLedger)
    #: Facts that declared a corpus-derivable check which could not execute — UNTESTED premises, like
    #: ``provisional_facts``, named in the headline beside them. Advisory: nothing downstream reads a
    #: fact that is not verified.
    unresolved_facts: list[str] = field(default_factory=list)
    #: Declared facts whose payload could not be read, so the scan ran WITHOUT them. Named in the
    #: headline beside the untested premises: a dropped fact holds nothing (one malformed
    #: declaration must not refuse the rest), but a headline claiming every declared check ran
    #: over a scan that never read one of them is false.
    dropped_facts: list[DroppedFact] = field(default_factory=list)
    #: Every scan gate that RAN for this report, finding or not — what separates an overrule that went
    #: ``stale`` from one whose gate never ran (``not_evaluated``).
    evaluated_gates: set[str] = field(default_factory=set)
    #: What each overrule given to :meth:`apply_overrules` did. Empty until it is called.
    overrule_outcomes: list[OverruleOutcome] = field(default_factory=list)

    @property
    def gate_findings(self) -> list[GateFinding]:
        """The findings the holding items mint, in item order. Built from the items, so there is one producer."""
        return [item.finding for item in self.items if item.finding is not None]

    def apply_overrules(self, overrules: Sequence[Overrule]) -> None:
        """Judge ``overrules`` against this report's findings. THE one place a scan overrule is applied.

        Only scan-gate overrules are this report's: a ledger carries every gate's, and the others are
        not decided here. They are selected by the ``scan.`` NAMESPACE, not by membership in the
        known scan gates, so a misspelled ``scan.*`` id reaches :func:`apply_overrules` and raises
        :class:`~plurai.gates.catalogue.UnknownGateError` — filtering it out here would leave it
        with no outcome at all, and the hold it was written to lift would stand with nothing
        saying why.

        An ``applied`` overrule releases its finding's hold — the item renders as an advisory, with no
        mention of why — and changes nothing else: the verdict, the evidence and every consumer of a
        verified-only fact stay exactly as they were.
        """
        mine = [overrule for overrule in overrules if overrule.gate_id.startswith(SCAN_GATE_PREFIX)]
        self.overrule_outcomes = apply_overrules(self.gate_findings, mine, evaluated_gates=self.evaluated_gates)

    def _lifted(self, item: ReportItem) -> bool:
        return item.finding is not None and item.finding.fingerprint in applied_fingerprints(self.overrule_outcomes)

    def _effective(self, item: ReportItem) -> ReportItem:
        """The item as it renders: a lifted hold is an advisory, and says nothing about the overrule."""
        return replace(item, severity=SEVERITY_ADVISORY) if item.severity == SEVERITY_FIX_FIRST and self._lifted(item) else item

    def of_severity(self, severity: str) -> list[ReportItem]:
        return [i for i in map(self._effective, self.items) if i.severity == severity]

    @property
    def fix_first(self) -> list[ReportItem]:
        """Held items — each a holding finding no overrule lifted — most severe first by the AUTHOR's declaration.

        Ordering matters because these render at the top and a reader acts on the first one;
        an unranked list can bury a `critical` contradiction under `low` ones.
        """
        return sorted(
            self.of_severity(SEVERITY_FIX_FIRST),
            key=lambda i: SEVERITY_ORDER.index(i.declared_severity) if i.declared_severity in SEVERITY_ORDER else len(SEVERITY_ORDER),
        )

    @property
    def proceed(self) -> bool:
        """False while any holding finding is un-overruled — i.e. while :attr:`fix_first` is non-empty,
        the ONE predicate the headline, the rendering and the exit code all read. Nothing else gates."""
        return not self.fix_first

    def gate_authority(self) -> GateAuthorityBlock:
        """The findings and what each overrule did. INTERNAL: never in :meth:`as_dict` (the published
        artifact) and never rendered by :meth:`render_markdown` — it carries the overrule reasons."""
        return GateAuthorityBlock(findings=self.gate_findings, overrules=list(self.overrule_outcomes))

    def gate_count_line(self, *, record_path: str | None = None) -> str | None:
        """The one ``[gate]`` terminal line the other gate-carrying commands print, or ``None`` when there
        is nothing to count (no finding, no overrule). Counts only — never a reason: the reasons are
        internal and live in the record at ``record_path``."""
        if not self.gate_findings and not self.overrule_outcomes:
            return None
        kinds = [o.outcome for o in self.overrule_outcomes]
        lifted = applied_fingerprints(self.overrule_outcomes)
        held = sum(1 for f in self.gate_findings if f.holds and f.fingerprint not in lifted)
        where = f'; see {record_path}' if record_path else ''
        return (
            f'[gate] {len(self.gate_findings)} finding(s), {held} held, {kinds.count("applied")} overruled, '
            f'{kinds.count("stale")} stale, {kinds.count("not_evaluated")} not evaluated{where}'
        )

    @property
    def contradicted_facts(self) -> list[str]:
        """Fact ids the corpus refuted, held or not. A lifted hold renders as an advisory, but the
        check still FOUND the conclusion false — so a PROCEED headline names these rather than
        claiming no check found one."""
        return [item.finding.subject for item in self.items if item.finding is not None and item.finding.gate_id == FACT_CONTRADICTED]

    @property
    def unexecuted_checks(self) -> list[str]:
        """Checks that could NOT run, by name. A PROCEED over a non-empty list is still a
        proceed — nothing was refuted — but it is not a clean bill."""
        return sorted(name for name, entry in self.checks.as_dict().items() if entry['status'] == 'blocked')

    def headline(self) -> str:
        """The first sentence a customer reads, stating WHAT HAPPENED rather than a verdict.

        Three states, not two, so that the summary line never asserts a completeness the body
        contradicts: a report with blocked checks renders ``[blocked]`` underneath, so the header
        must not read "every declared corpus-derivable check executed" — the not-run-is-not-a-pass
        rule holds for the sentence that summarises it. A PROCEED means NO CHECK REFUTED A CLAIM;
        whether every check could run is a separate fact, stated in the same sentence.
        """
        blocked = self.unexecuted_checks
        total = len(self.checks.as_dict())
        # Stated in the SAME sentence as the verdict, for the reason the blocked count is: a
        # verdict whose premises include facts nothing tested is not the clean bill it reads as,
        # and a defect in one of them surfaces only after generation, at the price of a regeneration.
        #
        # On BOTH branches. The FIX-FIRST branch is where the operator is LEAST likely to hear it
        # otherwise: they fix the blockers, see exit 0, and never learn the run still stands on N
        # untested premises — and the second pass, being clean, says nothing about them either
        # because the facts did not change.
        soft = (
            f' {len(self.provisional_facts)} fact(s) are ASSERTED, UNVERIFIED — no static route to check them '
            f'({", ".join(self.provisional_facts[:SAMPLE_LIMIT])}'
            f'{f", … (+{len(self.provisional_facts) - SAMPLE_LIMIT} more)" if len(self.provisional_facts) > SAMPLE_LIMIT else ""}), '
            'so whatever is generated from this scan is PROVISIONAL over them.'
            if self.provisional_facts
            else ''
        )
        if self.unresolved_facts:
            soft += (
                f' {len(self.unresolved_facts)} fact(s) declare a check that COULD NOT EXECUTE and are UNTESTED premises '
                f'({sampled_detail(self.unresolved_facts, pointer="the `fact_verification` section of scan_contract.json")}).'
            )
        if self.dropped_facts:
            # Named, never only counted, and on every branch: a dropped fact holds nothing (one
            # malformed declaration is dropped at a time, so the rest are still decided), so this
            # sentence is the only place the headline says the scan ran WITHOUT it.
            dropped_ids = [d.fact_id for d in self.dropped_facts]
            soft += (
                f' {len(dropped_ids)} declared fact(s) were DROPPED UNREAD — malformed, so no check ran on them '
                f'({sampled_detail(dropped_ids, pointer="the `fact_verification.dropped` section of scan_contract.json")}).'
            )
        if not self.proceed:
            return (
                f'**FIX FIRST — {len(self.fix_first)} holding finding(s).** Generation on a premise this scan refuted '
                'spends the qualification loop discovering that.' + soft
            )
        # A PROCEED does not mean nothing was refuted: a contradicted fact whose hold was released
        # still renders, as an advisory. Saying "none found" over it is false, so the count and the
        # fact ids take that sentence's place.
        contradicted = self.contradicted_facts
        refuted = (
            f'{len(contradicted)} scan conclusion(s) contradicted by the corpus '
            f'({sampled_detail(contradicted, pointer="the `fact_verification` section of scan_contract.json")}); '
            'each is listed with its evidence under Advisories below.'
            if contradicted
            else ''
        )
        if not blocked:
            lead = (
                f'**PROCEED** — all {total} check(s) executed, and found {refuted}'
                if contradicted
                else f'**PROCEED** — all {total} check(s) executed, and none found a scan conclusion false against the corpus.'
            )
            # Said ONLY when it is true: over a dropped fact, a declared check did NOT run, and the
            # soft text names it instead.
            return lead + ('' if self.dropped_facts else ' Every declared corpus-derivable check ran.') + soft
        return (
            f'**PROCEED — WITH {len(blocked)} OF {total} CHECK(S) NOT EXECUTED.** '
            + (f'The checks that ran found {refuted}' if contradicted else 'No check that ran found a scan conclusion false against the corpus.')
            + ' These could not run at all and therefore establish NOTHING: '
            f'{", ".join(blocked)}. Read their reasons under Checks below before treating this as a clean bill.' + soft
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'schema': self.schema,
            'proceed': self.proceed,
            # Beside ``proceed``, not derivable from it: a consumer reading the boolean alone
            # cannot tell a scan where everything ran from one where half the checks could not,
            # and the two license very different confidence in the same True.
            'unexecuted_checks': self.unexecuted_checks,
            # Beside them, and not folded into them: a consumer deciding whether to build on this
            # scan needs the facts nothing tested by NAME, so it can re-check the ones its own
            # generation is about to depend on.
            'provisional_facts': list(self.provisional_facts),
            'unresolved_facts': list(self.unresolved_facts),
            # By name, beside them: declared, never read, never decided. The reasons ride the
            # `fact_verification.dropped` section.
            'dropped_facts': [d.fact_id for d in self.dropped_facts],
            'headline': self.headline(),
            'fix_first_count': len(self.fix_first),
            'items': [i.as_dict() for i in map(self._effective, self.items)],
            'sections': self.sections,
            'checks': self.checks.as_dict(),
            # Every holding finding, held or lifted, with its fingerprint: what a caller overrules by.
            # No overrule rides here — this dict is published into scan_contract.json, which is
            # uploaded and committed; what each overrule did goes to the internal sidecar
            # (``contract.gate_authority_payload``) and nowhere else.
            'gate_findings': [f.model_dump(mode='json') for f in self.gate_findings],
        }

    def render_markdown(self) -> str:
        lines = ['# Pre-generation findings', '', self.headline(), '']
        if self.proceed and self.contradicted_facts:
            lines += ['This is not an all-clear: the contradicted conclusions named above are under Advisories, and every check below states what it did.', '']
        elif self.proceed:
            lines += ['This is an explicit all-clear about what WAS checked, not an absent report: every check below states what it did.', '']
        else:
            for item in self.fix_first:
                lines.append(f'- **[{item.check}] {item.summary}**')
                if item.detail:
                    lines.append(f'  - {item.detail}')
                if item.provenance:
                    lines.append(f'  - provenance: {", ".join(item.provenance)}')
            lines.append('')

        advisories = self.of_severity(SEVERITY_ADVISORY)
        if advisories:
            lines += ['## Advisories', '']
            lines += [f'- [{i.check}] {i.summary}' + (f' — {i.detail}' if i.detail else '') for i in advisories]
            lines.append('')

        info = self.of_severity(SEVERITY_INFO)
        if info:
            lines += ['## What the scan established', '']
            lines += [f'- [{i.check}] {i.summary}' + (f' — {i.detail}' if i.detail else '') for i in info]
            lines.append('')

        lines += ['## Checks', '']
        lines += [f'- `{name}` [{entry["status"]}]: {entry["detail"]}' for name, entry in self.checks.as_dict().items()]
        return '\n'.join(lines) + '\n'


def sampled_detail(values: Sequence[str], *, pointer: str, limit: int = SAMPLE_LIMIT) -> str:
    """The values joined when they fit; over ``limit``, the first few plus the count and the pointer.

    At or under the limit this is EXACTLY a plain join — no ``e.g.``, no ``+0 more``. The line
    already shows every entry, and hedging a complete list reads as though it were partial.
    """
    if len(values) <= limit:
        return ', '.join(values)
    return f'e.g. {", ".join(values[:limit])}, … (+{len(values) - limit} more; the complete list is {pointer})'


def build_pre_generation_report(
    *,
    verification: FactVerification,
    allowlist: UrlAllowlist,
    inventory: SourceInventory,
    probe_set: ProbeSet | None = None,
    smoke_probes: SmokeProbeSet | None = None,
    drift: DriftReport | None = None,
    corpus_id: str | None = None,
    load_warnings: Sequence[str] = (),
    citations: CitationAudit | None = None,
    governance: StateGovernance | None = None,
    services: ExternalServiceAudit | None = None,
    closed_class_decisions: Sequence[ClosedClassDecision] | None = None,
    unmatched_closed_class_overrides: Sequence[str] = (),
    dropped_facts: Sequence[DroppedFact],
    empty_roots: Sequence[EmptyRoot],
    declaration_problems: Sequence[DeclarationProblem],
) -> PreGenerationReport:
    """Aggregate the scan-contract artifacts into one report. No new computation.

    ``dropped_facts``, ``empty_roots`` and ``declaration_problems`` are the inputs the shell could
    not read whole: each is reported (an empty root HOLDS), and the rest of the scan is decided
    without it. Required, because a caller that forgets one reports a cleaner scan than it ran.
    """
    report = PreGenerationReport()

    # --- typed facts -------------------------------------------------------------------
    counts = verification.counts
    asserted = verification.unverified_of_kind(UNVERIFIED_NO_STATIC_ROUTE)
    unresolvable = verification.unverified_of_kind(UNVERIFIED_UNRESOLVABLE)
    report.provisional_facts = [v.fact.fact_id for v in asserted]
    report.unresolved_facts = [v.fact.fact_id for v in unresolvable]
    report.dropped_facts = list(dropped_facts)
    fact_detail = (
        f'{counts[VERIFIED]} verified / {counts[CONTRADICTED]} contradicted / '
        f'{len(verification.unverified_of_kind(UNVERIFIED_NOT_DERIVABLE))} not-derivable / '
        f'{len(asserted)} asserted-unverified / '
        f'{len(unresolvable)} unresolvable' + (f' / {len(dropped_facts)} dropped unread' if dropped_facts else '') + '. ' + verification.coverage_note()
    )
    if not verification.verdicts:
        report.checks.record(
            'typed_fact_verification',
            'flagged' if dropped_facts else 'blocked',
            f'every declared fact ({len(dropped_facts)}) was dropped unread, so this check decided nothing'
            if dropped_facts
            else 'no typed fact was declared for this scan, so this check decided nothing',
            finding_count=len(dropped_facts),
        )
    else:
        report.evaluated_gates.add(FACT_CONTRADICTED)
        if flagged := len(verification.flagged) + len(dropped_facts):
            report.checks.flagged('typed_fact_verification', flagged, fact_detail)
        else:
            report.checks.ran('typed_fact_verification', fact_detail)
    report.sections['fact_verification'] = {**verification.as_dict(), 'dropped': [d.as_dict() for d in dropped_facts]}
    for verdict in verification.contradicted:
        # HOLDS generation: the PRD still states the refuted claim to every generator. The evidence
        # carries the matched excerpts, so the caller can judge them — re-author the fact, or
        # overrule the hold. An overrule never flips the verdict: only a VERIFIED fact feeds a probe.
        report.items.append(
            ReportItem(
                severity=SEVERITY_FIX_FIRST,
                check='typed_fact_verification',
                summary=f'typed fact {verdict.fact.fact_id!r} is FALSE against the corpus',
                detail=f'{verdict.fact.claim.strip()} — {verdict.reason}',
                declared_severity=verdict.fact.severity,
                provenance=(f'origin:{verdict.fact.origin}', *(e.unit_id for e in verdict.evidence)),
                finding=make_finding(
                    FACT_CONTRADICTED,
                    subject=verdict.fact.fact_id,
                    evidence={
                        'fact_id': verdict.fact.fact_id,
                        'kind': verdict.fact.kind,
                        'check': verdict.fact.check_payload(),
                        'scopes_addressed': verdict.scopes_addressed,
                        'matches': [e.as_dict() for e in verdict.evidence],
                    },
                    observed={'claim': verdict.fact.claim, 'origin': verdict.fact.origin},
                    remedy=f'{verdict.fact.claim.strip()} — {verdict.reason}',
                    draft_ref=None,
                ),
            )
        )
    for verdict in unresolvable:
        # ADVISORY. The author declared a corpus-derivable check and it could not execute, so the
        # fact is an UNTESTED premise — named in the headline beside the asserted ones. It holds
        # nothing because nothing downstream consumes a fact that is not verified.
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='typed_fact_verification',
                summary=f'fact {verdict.fact.fact_id!r} declares a check that COULD NOT EXECUTE',
                detail=verdict.reason,
                declared_severity=verdict.fact.severity,
                provenance=(f'origin:{verdict.fact.origin}',),
            )
        )
    for dropped in dropped_facts:
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='typed_fact_verification',
                summary=f'fact {dropped.fact_id!r} was DROPPED unread — its declaration is malformed, so the scan ran without it',
                detail=dropped.reason,
            )
        )
    for verdict in verification.unverified_of_kind(UNVERIFIED_NOT_DERIVABLE):
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='typed_fact_verification',
                summary=f'fact {verdict.fact.fact_id!r} is {UNVERIFIED_AT_SCAN} — behavioural, not corpus-derivable',
                detail=verdict.reason,
                provenance=(f'origin:{verdict.fact.origin}',),
            )
        )
    for verdict in asserted:
        # ADVISORY, not fix-first: the author is entitled to state a fact the corpus cannot
        # settle, and blocking on it would make the honest declaration the expensive one. What it
        # must never do is disappear: a fact dropped from the document entirely would make a
        # report over some of the declared facts render like one over all of them.
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='typed_fact_verification',
                summary=f'fact {verdict.fact.fact_id!r} is ASSERTED, UNVERIFIED — no static route to check this',
                detail=verdict.reason,
                declared_severity=verdict.fact.severity,
                provenance=(f'origin:{verdict.fact.origin}', *((f'source:{verdict.fact.source}',) if verdict.fact.source else ())),
            )
        )
    if verified := verification.of_status(VERIFIED):
        report.items.append(
            ReportItem(
                severity=SEVERITY_INFO,
                check='typed_fact_verification',
                summary=f'{len(verified)} fact(s) verified against the corpus',
                detail=', '.join(v.fact.fact_id for v in verified),
            )
        )

    # --- fact citations ----------------------------------------------------------------
    if citations is None or not citations.checked:
        report.checks.blocked(
            'fact_source_citations', citations.blocked_reason if citations else 'no source root was supplied, so no `source:` citation was resolved'
        )
    else:
        report.sections['fact_citations'] = citations.as_dict()
        defective_citations = citations.defective
        # BLOCKED, not `ran`, when the audit executed and established nothing. `subject` is what a
        # `path:LINE` citation is checked against, and a facts document written by hand, or one
        # without the field, carries none — so every line citation comes back UNCHECKED while the
        # summary still reads like an audit that ran. That is the off-by-N check this exists for,
        # silently off, under a headline that names no blocked check.
        if unestablished := citations.unestablished_reason():
            report.checks.blocked('fact_source_citations', unestablished)
        else:
            report.checks.record(
                'fact_source_citations', 'flagged' if defective_citations else 'ran', citations.summary(), finding_count=len(defective_citations)
            )
        for citation in defective_citations:
            # ADVISORY. A claim whose evidence is not where the author said it is has not been
            # checked by anyone — provenance the author should correct — but it holds nothing: the
            # whole audit rides the artifact's `fact_citations` section.
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='fact_source_citations',
                    summary=f'fact {citation.fact_id!r} cites evidence that does not resolve ({citation.status})',
                    detail=citation.reason,
                    provenance=((f'source:{citation.source}',) if citation.source else ()),
                )
            )
        if unresolvable := [v for v in citations.verdicts if v.status == CITATION_NOT_A_PATH]:
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='fact_source_citations',
                    summary=f'{len(unresolvable)} citation(s) are not filesystem paths, so this check decided nothing about them',
                    detail=sampled_detail([f'{v.fact_id} → {v.source}' for v in unresolvable], pointer='the `fact_citations` section of scan_contract.json'),
                )
            )
        if undecided := [v for v in citations.verdicts if v.status in (CITATION_UNCHECKED, CITATION_UNSUPPORTED)]:
            # ADVISORY, and it must exist. These citations resolved to a file and were NOT
            # established: the line could not be read, or the subject is nowhere in it. Neither
            # blocks — a heuristic window must not stop a build, and an absence-shaped claim cites
            # exactly the place its subject is not — but leaving them out of the document is how a
            # report over the checkable half reads like a report over all of it.
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='fact_source_citations',
                    summary=f'{len(undecided)} citation(s) resolved to a file without establishing the LINE they name',
                    detail=sampled_detail(
                        [f'{v.fact_id} ({v.status}) — {v.reason}' for v in undecided], pointer='the `fact_citations` section of scan_contract.json'
                    ),
                )
            )
        if resolved := [v for v in citations.verdicts if v.status == CITATION_RESOLVED]:
            # Named for what was actually established. A bare `N of M resolved` would read "…and the
            # line" over verdicts where no line was read, so the count says how many lines were read.
            lines_read = sum(1 for v in resolved if v.line_checked)
            report.items.append(
                ReportItem(
                    severity=SEVERITY_INFO,
                    check='fact_source_citations',
                    summary=f'{len(resolved)} fact citation(s) resolve to the evidence they name, {lines_read} with the cited LINE read for its subject',
                    detail=sampled_detail([v.fact_id for v in resolved], pointer='the `fact_citations` section of scan_contract.json'),
                )
            )

    # --- URL allowlist -----------------------------------------------------------------
    if len(allowlist):
        report.checks.ran(
            'placeholder_url_compilation',
            f'{len(allowlist)} URL(s) derived from the corpus, {len(allowlist.templates)} of them placeholder TEMPLATES kept as patterns',
        )
    else:
        report.checks.blocked(
            'placeholder_url_compilation',
            'the URL extractor found no http(s) token in the scanned corpus. That is not proof the corpus documents no '
            'address: a scheme-less "www." form, or a path carrying ()[]{}, is one the extractor does not see',
        )
    # A SUMMARY, never a second copy. The complete set is the artifact's top-level
    # ``url_allowlist``, and that is the only copy anything is entitled to read: a catalog corpus
    # licenses enough URLs that a second embedding costs hundreds of KB of one artifact for bytes
    # no consumer opens. The pointer names the canonical key, so the summary can never be
    # mistaken for the whole set.
    report.sections['url_allowlist'] = {
        'count': len(allowlist),
        'template_count': len(allowlist.templates),
        'literal_count': len(allowlist.literals),
        # EXPLICIT None, never omitted and never []. A reader that does
        # `.get('urls', [])` reads an empty list as "no URL is licensed" — the exact
        # false-absence this artifact exists to prevent, since a documented link the set never
        # registers is then graded as fabricated. `None` makes such a reader trip instead.
        'urls': None,
        'urls_published_at': f'{ALLOWLIST_URLS_KEY} (top level of scan_contract.json)',
    }
    for entry in allowlist.templates:
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='placeholder_url_compilation',
                summary=f'{entry.template} is a TEMPLATE, licensed as a pattern rather than a literal',
                detail=entry.decision,
                provenance=entry.provenance,
            )
        )
    if literals := allowlist.literals:
        report.items.append(
            ReportItem(
                severity=SEVERITY_INFO,
                check='placeholder_url_compilation',
                summary=f'{len(literals)} literal URL(s) licensed exactly as written',
                detail=sampled_detail([e.template for e in literals], pointer=ALLOWLIST_POINTER),
            )
        )

    # --- corpus sources ----------------------------------------------------------------
    report.checks.ran('corpus_source_hash', f'{len(inventory.files)} source file(s) hashed; content_hash {inventory.content_hash[:16] or "(none)"}')
    report.sections['corpus_sources'] = {**inventory.as_dict(), 'corpus_id': corpus_id}
    report.items.append(
        ReportItem(
            severity=SEVERITY_INFO,
            check='corpus_source_hash',
            summary=f'corpus scope is a content hash, not a date: {len(inventory.files)} file(s), {inventory.content_hash[:16]}',
            detail=f'stored beside corpus_id={corpus_id}' if corpus_id else 'no corpus_id supplied; the hash stands alone until one is recorded',
        )
    )
    if drift is None:
        report.checks.blocked(
            'corpus_source_drift', 'no recorded inventory was supplied to compare against (the recording pass, or a caller that did not pass one)'
        )
    else:
        report.sections['corpus_source_drift'] = drift.as_dict()
        if drift.blocked_reason:
            # BLOCKED, not flagged-with-zero-findings. `blocked` is the ledger's word for "the
            # input this check needs was absent", which is exactly what blocked_reason means;
            # flagged(0) reads as "looked, found nothing" to a consumer counting findings.
            report.checks.blocked('corpus_source_drift', drift.blocked_reason)
        elif drift.matched:
            report.checks.ran('corpus_source_drift', drift.summary())
        else:
            report.checks.flagged('corpus_source_drift', 1, drift.summary())
        if not drift.matched:
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='corpus_source_drift',
                    summary=drift.summary(),
                    detail=(
                        'ADVISORY by design: a customer editing their own documentation is normal. It is recorded so '
                        'a suite scored against moved sources says so.'
                    ),
                )
            )

    # --- probes ------------------------------------------------------------------------
    if probe_set is None:
        report.checks.blocked('absence_probes', 'no probe derivation was requested for this scan')
    else:
        targets = suppression_targets(verification)
        report.checks.record(
            'absence_probes', 'ran' if probe_set.probes else 'blocked', probe_budget_note(probe_set, targets), finding_count=len(probe_set.probes)
        )
        report.sections['absence_probes'] = {**probe_set.as_dict(), 'suppression_still_applies_to': list(targets)}
        for probe in probe_set.probes:
            report.items.append(
                ReportItem(
                    severity=SEVERITY_INFO,
                    check='absence_probes',
                    summary=f'probe {probe.policy_name} asks for the verified gap {probe.fact_id!r}',
                    detail=probe.ask,
                    provenance=(f'fact:{probe.fact_id}',),
                )
            )
        for skipped in probe_set.unprobed:
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='absence_probes',
                    summary=f'absence fact {skipped.fact_id!r} is UNPROBED',
                    detail=skipped.reason,
                )
            )

    # --- scan-time smoke probes --------------------------------------------------------
    if smoke_probes is None:
        report.checks.blocked('scan_time_smoke_probes', 'no smoke-probe derivation was requested for this scan')
    else:
        # BLOCKED as this report is BUILT — derivation alone sends nothing to an agent, whatever
        # consent says, and the two execution_status wordings differ while the ledger VALUE does
        # not, so consent alone can never read as having changed what ran.
        #
        # NOT the final word on the entry. A caller that actually FIRES the probes (a live agent
        # handle plus consent, via the CLI's execution lane) overwrites this entry afterwards with
        # what the run did. Read the published ledger, never this line, for whether anything left
        # the machine.
        report.checks.blocked('scan_time_smoke_probes', smoke_probes.execution_status)
        report.sections['scan_time_smoke_probes'] = smoke_probes.as_dict()
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='scan_time_smoke_probes',
                summary=f'{len(smoke_probes.probes)} cheap live probe(s) derived from this scan, none executed',
                detail=(f'{smoke_probes.execution_status} Probe transcripts would be evidence about the LIVE agent; nothing in THIS ITEM is that evidence.'),
            )
        )

    # --- agent_context governance ------------------------------------------------------
    if governance is None:
        report.checks.blocked('agent_context_governance', 'no state schema and no stateless finding reached this check')
    else:
        report.sections['agent_context_governance'] = governance.as_dict()
        if governance.decided:
            report.checks.ran('agent_context_governance', governance.detail)
            report.items.append(
                ReportItem(severity=SEVERITY_INFO, check='agent_context_governance', summary=f'agent_context is {governance.status}', detail=governance.detail)
            )
        else:
            # BLOCKED, and an ADVISORY rather than fix-first. The schema is an env-gen OUTPUT, so
            # at scan time its absence is normal and hard-failing on it would block every scan.
            # What it must not do is read as verified-empty: blocked puts it in the headline's
            # not-executed list, where the operator sees it before generation spends.
            report.checks.blocked('agent_context_governance', governance.detail)
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='agent_context_governance',
                    summary=f'agent_context is {STATE_UNGOVERNED} — not verified-empty',
                    detail=governance.detail,
                )
            )

    # --- "no external services" vs the loaded env -------------------------------------
    if services is None or not services.checked:
        report.checks.blocked(
            'external_service_claims',
            services.blocked_reason if services else 'no env or config file reached this check, so no absence-of-service claim was compared against one',
        )
    else:
        report.sections['external_service_claims'] = services.as_dict()
        contradicted = services.contradicted
        report.checks.record('external_service_claims', 'flagged' if contradicted else 'ran', services.summary(), finding_count=len(contradicted))
        for claim in contradicted:
            # ADVISORY, not fix-first, and this is the one place the two could reasonably differ.
            # It stays advisory because every input to it is circumstantial: the env file in the
            # tree may be a developer's rather than the deployment's, a configured service may be
            # unreachable at runtime, and the claim's own scope is a sentence this check reads
            # rather than a predicate it executes. A build stopped by any of those is a build
            # stopped by a heuristic, which is the bar `check_complication_coverage` set.
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='external_service_claims',
                    summary=f'fact {claim.fact_id!r} claims a service is absent, and the loaded env CONFIGURES one',
                    detail=f'{claim.claim} — '
                    + ', '.join(f'{e.variable} in {e.source} declares {e.service}' for e in claim.contradicted_by)
                    + '. Either narrow the claim to what the deployment actually disables, or turn the variable off; a scenario '
                    'generated from this premise will assume a surface the agent has.',
                    provenance=tuple(dict.fromkeys(e.source for e in claim.contradicted_by)),
                )
            )
        if services.unclassified:
            # The OTHER half that keeps a clean verdict honest: the table is closed over a fixed set
            # of vendors, so a service it cannot name (an observability vendor, a hosted model API
            # or an in-house billing API) matches no indicator and the audit returns
            # `checked=True, contradicted=()`. Under-reporting is fine — delivering it as coverage
            # is not, so the count of what the table could not name is a document-level item and
            # not only a clause inside the summary line.
            #
            # ADVISORY, and never anything else. "Ends in _API_KEY" names no service and refutes
            # no sentence; filing it as a finding would spend exactly the false-positive
            # discipline that makes the CONTRADICTED items above worth reading.
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='external_service_claims',
                    summary=f'{len(services.unclassified)} configured-looking env variable(s) name no service this check knows — it decided NOTHING about them',
                    detail=sampled_detail(
                        [f'{u.variable} in {u.source}' for u in services.unclassified],
                        pointer='the `external_service_claims` section of scan_contract.json',
                    )
                    + '. The service table is closed, so an unlisted vendor is ABSENT from the verdict above rather than cleared '
                    'by it. Confirm each is local, or declare it with --service-indicators so the next scan can decide it.',
                    provenance=tuple(dict.fromkeys(u.source for u in services.unclassified)),
                )
            )
        if services.unreadable:
            # The half that keeps a clean verdict honest: a variable in a file nobody read is
            # invisible to the loop above, and "no service is configured" over a partial read is
            # the same sentence as over a complete one.
            report.items.append(
                ReportItem(
                    severity=SEVERITY_ADVISORY,
                    check='external_service_claims',
                    summary=f'{len(services.unreadable)} env file(s) exist and could NOT be read, so the comparison ran over part of the configuration',
                    detail=sampled_detail(list(services.unreadable), pointer='the `external_service_claims` section of scan_contract.json'),
                )
            )

    # --- degraded inputs, and inputs never read at all -----------------------------------
    # TWO populations under one check, deliberately: a source admitted in a damaged form and a
    # source the walk never admitted are the same finding from the reader's side — the corpus
    # every fact was decided over is not the corpus on disk. The second population is the files
    # the loader skips by SUFFIX: left uncounted, one README.md beside many .html help pages
    # renders exactly like a corpus of one file.
    report.evaluated_gates.add(CORPUS_ROOT_EMPTY)
    report.checks.record(
        'corpus_load',
        'flagged' if load_warnings or empty_roots else 'ran',
        f'{len(inventory.files)} source(s) read, {len(load_warnings)} warning(s) — sources admitted in a degraded form, or not read at all'
        + (f'; {len(empty_roots)} corpus root(s) admitted NO file' if empty_roots else ''),
        finding_count=len(load_warnings) + len(empty_roots),
    )
    for empty in empty_roots:
        # HOLDS generation: every fact scoped to this root is decided over nothing. The census of
        # what the root DID hold says which --suffixes would admit it.
        skipped = ', '.join(f'{count} {suffix}' for suffix, count in empty.skipped_by_suffix.items()) or 'no file with a suffix at all'
        remedy = (
            f'{empty.root} holds {skipped}; this scan admits {", ".join(empty.admitted_suffixes)}. Pass --suffixes for the '
            'extension that holds the knowledge, or drop the root.'
        )
        report.items.append(
            ReportItem(
                severity=SEVERITY_FIX_FIRST,
                check='corpus_load',
                summary=f'corpus root {empty.root.name!r} admitted NO file, so nothing scoped to it was decided',
                detail=remedy,
                provenance=(str(empty.root),),
                finding=make_finding(
                    CORPUS_ROOT_EMPTY,
                    subject=empty.root.name,
                    evidence={'root': empty.root.name, 'admitted_suffixes': list(empty.admitted_suffixes), 'skipped_by_suffix': dict(empty.skipped_by_suffix)},
                    observed={'path': str(empty.root)},
                    remedy=remedy,
                    draft_ref=None,
                ),
            )
        )
    for warning in load_warnings:
        report.items.append(
            ReportItem(severity=SEVERITY_ADVISORY, check='corpus_load', summary='a source was NOT read, or was read in a DEGRADED form', detail=warning)
        )
    for problem in declaration_problems:
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check=problem.check,
                summary=f'a `{problem.check}:` declaration could not be read, so this scan ran WITHOUT it',
                detail=problem.detail,
            )
        )

    _report_closed_class_detectors(report, closed_class_decisions, unmatched_closed_class_overrides)
    return report


def _report_closed_class_detectors(
    report: PreGenerationReport, decisions: Sequence[ClosedClassDecision] | None, unmatched_overrides: Sequence[str] = ()
) -> None:
    """Whether every closed set that REQUIRES a code-side detector actually got one.

    Three severities, because the gate is a prompt and not a wall, and the three states are
    genuinely different claims about the customer's own document:

    * ``required: true`` with no active detector — **fix-first**. The author said this class must
      be checked in code and nothing checks it, so proceeding ships a licence a judge applies
      alone over a set the author did not trust a judge with.
    * ``required: false`` — **info**. A closed class with no code check is a legitimate choice; it
      just has to be a stated one, and the report and the judge prompt both say so.
    * no ``detector:`` block at all — **advisory**. Nobody said. This is the state every existing
      PRD is in, so blocking on it would refuse documents that were correct when they were
      written; naming it is what gets the declaration added.

    An ACTIVE detector is also reported, as info, because "0 findings" over a check that decided
    nothing reads exactly like "0 findings" over a check that measured six classes and passed.
    """
    for subject in unmatched_overrides:
        # ADVISORY, and never silent. The customer wrote a detector for a subject no `closed_set`
        # fact declares, so it was never applied — a misspelled subject is invisible without this.
        # It holds nothing: an override that applied to no fact changed no verdict.
        report.items.append(
            ReportItem(
                severity=SEVERITY_ADVISORY,
                check='closed_class_detectors',
                summary=f'`closed_class_detectors` declares a detector for {subject!r}, which no `closed_set` fact names',
                detail='The declaration is joined to a fact by SUBJECT, so this one was never applied to anything and no gate '
                'ever ran against it. Correct the subject to match a `data_profile` / `closed_set` fact in simulation_prd.md, '
                'or remove the declaration.',
            )
        )
    if decisions is None:
        report.checks.blocked(
            'closed_class_detectors', 'no corpus units and no typed facts reached this scan, so no closed set could be measured for a detector'
        )
        return
    report.evaluated_gates.add(CLOSED_CLASS_DETECTOR_MISSING)
    closed_sets = list(decisions)
    if not closed_sets:
        report.checks.record(
            'closed_class_detectors',
            'flagged' if unmatched_overrides else 'blocked',
            f'{len(unmatched_overrides)} declared override(s) match no `closed_set` fact'
            if unmatched_overrides
            else 'this scan declares no `data_profile` / `closed_set` fact, so there was no closed class to check',
            finding_count=len(unmatched_overrides),
        )
        return
    missing = [d for d in closed_sets if d.required and not d.active]
    active = [d for d in closed_sets if d.active]
    report.checks.record(
        'closed_class_detectors',
        'flagged' if missing or unmatched_overrides else 'ran',
        f'{len(closed_sets)} closed set(s): {len(active)} have a detector ACTIVE in code, {len(missing)} REQUIRE one and have none, '
        f'{sum(1 for d in closed_sets if not d.declared)} say nothing about whether one is required',
        finding_count=len(missing) + len(unmatched_overrides),
    )
    for decision in closed_sets:
        report.items.append(_closed_class_item(decision))


def _closed_class_item(decision: ClosedClassDecision) -> ReportItem:
    """One line per closed set, with the ladder's whole descent when it refused."""
    fact = decision.fact
    provenance = (f'origin:{fact.origin}', *((fact.source,) if fact.source else ()))
    head = f'fact {fact.fact_id!r} (data_profile.closed_set, subject={fact.subject or "(none)"})'
    if decision.active and decision.record is not None:
        rule = str((decision.record.get('detector') or {}).get('rule') or '')
        return ReportItem(
            severity=SEVERITY_INFO,
            check='closed_class_detectors',
            summary=f'{head} is checked in code by {rule}',
            detail=f'{len(decision.record["members"])} member(s) lifted from {len(decision.scopes)} corpus field(s); every gate passed.',
            provenance=provenance,
        )
    if decision.required:
        # HOLDS generation: `required: true` is the author's own request to hold. Lifting it means
        # "proceed judge-only" — the record stays exactly what the ladder decided.
        #
        # The generic remedy is appended only where it APPLIES. On an oversized class "add members"
        # is the opposite of the fix, and its own descent already names the one that works — a
        # remedy that does not fit is a remedy the reader learns to skip.
        detail = '  ladder:\n' + decision.descent() + ('' if decision.oversized else '\n' + GATE_REMEDY)
        return ReportItem(
            severity=SEVERITY_FIX_FIRST,
            check='closed_class_detectors',
            summary=f'{head} REQUIRES a deterministic detector and none passed',
            detail=detail,
            declared_severity=fact.severity,
            provenance=provenance,
            finding=make_finding(
                CLOSED_CLASS_DETECTOR_MISSING,
                subject=fact.fact_id,
                evidence={
                    'fact_id': fact.fact_id,
                    'subject': fact.subject,
                    'member_count': len(decision.record.get('members') or ()) if decision.record is not None else 0,
                    'oversized': decision.oversized,
                    'descent': decision.descent(),
                },
                observed={'provenance': list(provenance)},
                remedy=detail,
                draft_ref=None,
            ),
        )
    if decision.declared:
        return ReportItem(
            severity=SEVERITY_INFO,
            check='closed_class_detectors',
            summary=f'{head} is CLOSED, JUDGE-ONLY by declaration (detector.required: false)',
            detail='No code check decides membership in this class. The judge is told so in as many words, and its verdict '
            'about a value outside the list is its own reading rather than a measurement.\n  ladder (run anyway, so the '
            'choice is informed — and so a DECLARED pattern that lost a gate is not read as a choice):\n' + decision.descent(),
            provenance=provenance,
        )
    return ReportItem(
        severity=SEVERITY_ADVISORY,
        check='closed_class_detectors',
        summary=f'{head} declares no `detector:` block — say `required: true` or `required: false`',
        detail='A closed set with nothing said about checking it is a licence a judge applies alone, with nobody having '
        'chosen that. Declaring `required: false` is a legitimate answer and makes the '
        'choice visible in this report and in the judge prompt.\n  ladder (run anyway, so the choice is informed):\n' + decision.descent(),
        provenance=provenance,
    )
