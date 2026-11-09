"""One call that runs the whole scan contract and returns its report.

The checks are separate modules because they answer separate questions, but they are ALWAYS run
together: an allowlist derived without verifying the facts that scoped it, or facts verified
against a corpus nobody hashed, is half a contract. This is the seam every caller uses — the
CLI, and any backend job that later ingests a scan.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plurai.scan_contract._dialog_urls import extract_urls
from plurai.gates.contract import Overrule
from plurai.scan_contract.citations import CitationAudit, audit_citations, cited_lines
from plurai.scan_contract.closed_class_gate import (
    CLOSED_CLASSES_KEY,
    ClosedClassDecision,
    decide_closed_class_detectors,
    declared_reply_scripts,
)
from plurai.scan_contract.corpus import CorpusUnit, DriftReport, EmptyRoot, SourceInventory, check_source_drift
from plurai.scan_contract.facts import (
    CONTRADICTED,
    KIND_ABSENCE,
    PREDICATE_CLOSED_SET,
    PREDICATE_INSTRUCTS,
    VERIFIED,
    DroppedFact,
    FactVerification,
    ScanFact,
    artifact_fact_verdicts,
    fact_id_is_positional,
    load_scan_facts,
    verify_facts,
)
from plurai.scan_contract.inputs import StateGovernance, check_agent_context_governance, load_variables
from plurai.scan_contract.probes import DEFAULT_PROBE_BUDGET, ProbeSet, derive_absence_probes
from plurai.scan_contract.render_actions import RENDER_ACTIONS_KEY, RENDER_ACTIONS_UNREADABLE, RenderSurface
from plurai.scan_contract.smoke_probes import derive_smoke_probes
from plurai.scan_contract.report import DeclarationProblem, PreGenerationReport, build_pre_generation_report
from plurai.scan_contract.services import ExternalServiceAudit, check_external_services
from plurai.scan_contract.urls import UrlAllowlist, derive_url_allowlist


@dataclass(frozen=True)
class ScanContractResult:
    """Everything the contract established, plus the one report a human reads."""

    report: PreGenerationReport
    verification: FactVerification
    #: These two are also serialized inside ``report.sections``. Kept as typed fields anyway,
    #: deliberately: ``sections`` is the wire view a consumer reads, and pulling a
    #: :class:`UrlAllowlist` back out of a dict to call ``licensing`` on it would make every
    #: caller re-parse what this object already holds.
    allowlist: UrlAllowlist
    inventory: SourceInventory
    probe_set: ProbeSet
    drift: DriftReport | None
    #: Where each fact says it was read, resolved against the scanned tree.
    citations: CitationAudit
    #: Whether ``agent_context`` is governed, and by what.
    governance: StateGovernance
    #: Whether any "no external services" claim survives the env the scan can read.
    services: ExternalServiceAudit
    #: The built corpus this scan's source fingerprint is stored beside, when one is known.
    #: Carried on the RESULT so ``artifact_payload`` cannot be handed a different one than the
    #: report already embeds.
    corpus_id: str | None = None
    #: Subjects the customer declared a detector for that no ``closed_set`` fact names — never
    #: applied, and invisible until it is reported.
    unmatched_closed_class_overrides: tuple[str, ...] = ()
    #: What the closed-class detector ladder decided for every ``closed_set`` fact, and why.
    #: Typed on the RESULT for the reason ``render_surface`` is: the artifact publishes what the
    #: contract was RUN with, and a consumer reading the report must be able to reach the same
    #: records the check filed its findings from.
    closed_class_decisions: tuple[ClosedClassDecision, ...] = ()
    #: The agent's declared render / action surface, as the caller discovered it under the
    #: source roots — ``None`` when no ``plurai.config.yaml`` declared one, which is a different
    #: fact from a surface with no verbs. Carried on the RESULT for the reason ``corpus_id`` is:
    #: ``artifact_payload`` emits what the contract was RUN with, and cannot be handed another.
    render_surface: RenderSurface | None = None
    #: Why a ``render_actions:`` declaration that EXISTS could not be read — the shell's
    #: ``declaration_problem`` for it — so the artifact says "present and unreadable" rather than
    #: carrying no field, which every reader takes for "nobody declared one".
    render_surface_unreadable: str | None = None

    def blocking_message(self) -> str:
        """The held findings, printed — each with the gate id and fingerprint an overrule binds to.

        Built from the REPORT's own fix-first list, the single signal ``proceed`` reads, so a
        held finding can never print here while the exit code says proceed, or the reverse.
        """
        items = self.report.fix_first
        lines = [
            f'{len(items)} finding(s) hold generation. These are inputs to generation, so proceeding spends the '
            'qualification loop on premises this scan refuted or could not decide:'
        ]
        for item in items:
            lines.append(f'  - [{item.check}] {item.summary}')
            if item.detail:
                lines.append(f'      {item.detail}')
            if item.provenance:
                lines.append(f'      provenance: {", ".join(item.provenance)}')
            if item.finding is not None:
                lines.append(f'      gate {item.finding.gate_id} [fingerprint {item.finding.fingerprint}]')
        return '\n'.join(lines)


def declared_agent_language(variables: Mapping[str, Any] | None) -> str | None:
    """The ONE language a ``scenario_variables.yaml`` declares its agent works in, or ``None``.

    ``ScenarioVariables.language`` has three shapes, all legal: a single value (the whole set is
    authored in it), a list (spread evenly), and a weight map (spread in proportion). Only the
    first names A language for the agent — a set deliberately mixing Hebrew and English has no
    single one, and picking the heaviest would tell the smoke derivation that the English
    placeholder ask is safe for a set that is 40% Hebrew.

    So a mix resolves to ``None``, which is the same answer an absent declaration gives and means
    the same thing here: fall back to the evidence in the derived asks themselves. Read off the raw
    mapping rather than the typed model because this package is vendored into the client SDK and
    may not import the generator.
    """
    if not isinstance(variables, Mapping):
        return None
    declared = variables.get('language')
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    if isinstance(declared, (list, tuple)):
        distinct = {str(entry).strip() for entry in declared if str(entry).strip()}
        return distinct.pop() if len(distinct) == 1 else None
    if isinstance(declared, Mapping):
        named = {str(key).strip() for key, weight in declared.items() if str(key).strip() and weight}
        return named.pop() if len(named) == 1 else None
    return None


def run_scan_contract(
    *,
    units: Sequence[CorpusUnit],
    inventory: SourceInventory,
    facts: Iterable[ScanFact],
    recorded_inventory: SourceInventory | None = None,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
    corpus_id: str | None = None,
    load_warnings: Sequence[str] = (),
    probe_consent: bool = False,
    source_roots: Sequence[Path] = (),
    state_schema_yaml: str | None = None,
    stateless_finding: str = '',
    variables_yaml: str | None = None,
    service_indicators: Mapping[str, Sequence[str]] | None = None,
    language: str | None = None,
    render_surface: RenderSurface | None = None,
    closed_class_overrides: Mapping[str, str] | None = None,
    overrules: Sequence[Overrule],
    dropped_facts: Sequence[DroppedFact],
    empty_roots: Sequence[EmptyRoot],
    declaration_problems: Sequence[DeclarationProblem],
) -> ScanContractResult:
    """Run every check and aggregate. Never raises on a finding: ``report.proceed`` is the one signal,
    and it is False while a holding finding is un-overruled.

    ``overrules`` are applied HERE, inside the checker, so both shells lift exactly the same holds
    (:meth:`~plurai.scan_contract.report.PreGenerationReport.apply_overrules`). ``dropped_facts``,
    ``empty_roots`` and ``declaration_problems`` are what the shell could not read whole — each is
    reported and the rest of the scan is decided without it. All four are REQUIRED: a default here
    would let a shell forget one and report a cleaner scan than it ran.

    ``render_surface`` is the agent's declared render / action surface, discovered by the caller
    under the SAME source roots the citations resolve against
    (:func:`plurai.scan_contract.render_actions.discover_render_surface`) — discovered there and
    not here because the shell owns the config reading; a malformed declaration arrives as a
    ``declaration_problem`` and the surface as ``None``. It is CARRIED, not checked: the check runs
    on the backend, over the agent's replies, and reads the surface off the artifact.

    The inputs after ``probe_consent`` are the scan's OWN artifacts rather than the customer's
    corpus: the tree the facts cite, the state schema (or the finding that stands in for it) and
    the typed variables. Each is optional and each absent one becomes a BLOCKED check with its
    reason — never a check that quietly passed.

    ``service_indicators`` is the caller's extension of the closed service table (see
    :func:`plurai.scan_contract.services.load_service_indicators`). It CLOSES a blind spot; it
    never hides one — the audit publishes the size of whatever table decided it either way.

    ``language`` is the language the agent works in. Passed explicitly by a caller that knows it,
    otherwise DERIVED from the supplied ``scenario_variables.yaml`` (:func:`declared_agent_language`)
    — which is where a customer declares it and which this function already parses. It is not
    cosmetic: the PLACEHOLDER smoke ask is an English sentence synthesized in
    :func:`plurai.scan_contract.smoke_probes.derive_smoke_probes`, and without a language that
    withholding decision falls back to the SCRIPT of the derived asks, which reads Spanish, French
    and Portuguese as English.
    """
    # Materialized FIRST: ``facts`` is an Iterable, and two consumers over a generator leave the
    # second one looking at an empty scan.
    declared = list(facts)
    verification = verify_facts(declared, units)
    citations = audit_citations(declared, source_roots)
    governance = check_agent_context_governance(state_schema_yaml, stateless_finding)
    variables, _ = load_variables(variables_yaml)
    # Over the SAME roots the citations resolve against: the ``.env`` that refutes a section-7
    # claim is a file the scan already had open, which is the whole reason this is cheap.
    services = check_external_services(declared, source_roots, indicators=service_indicators)
    # The agent's context is its knowledge AND its instructions: a URL a rule tells it to give is licensed by that rule —
    # read off the cited line(s), the bytes the agent is given, never off the PRD's quote of them.
    rule_urls = [
        (url, f'rule:{fact.fact_id}') for fact in declared if fact.predicate == PREDICATE_INSTRUCTS for url in extract_urls(cited_lines(fact, source_roots))
    ]
    allowlist = derive_url_allowlist(units, extra_licensed=rule_urls)
    probe_set = derive_absence_probes(verification, budget=probe_budget)
    smoke_probes = derive_smoke_probes(
        probe_set,
        [e.template for e in allowlist.templates],
        probe_consent=probe_consent,
        language=language or declared_agent_language(variables),
    )
    drift = check_source_drift(recorded_inventory, inventory) if recorded_inventory is not None else None
    # The closed-class detector ladder, over the SCANNED units. Run here and not on
    # the backend alone because this is the side that holds the customer's own text: they learn a
    # pattern is too loose BEFORE paying for a corpus build, and the same vendored code runs there
    # over the registry, so the two verdicts are comparable rather than merely similar.
    #
    # ``closed_class_overrides`` is the customer's ``plurai.config.yaml`` declaration, passed in as
    # a TYPED argument so every shell that calls this function writes the artifact identically.
    # Declarations go IN through this function; verdicts come out on the artifact —
    # hand-editing a requirement into the artifact is the drift the scan sweeps against.
    closed_class_decisions = (
        decide_closed_class_detectors(
            declared,
            units,
            overrides=closed_class_overrides,
            reply_scripts=declared_reply_scripts(language or declared_agent_language(variables)),
        )
        if units
        else None
    )
    # ONE join rule, computed once, so the report and the result cannot disagree about which
    # declarations were applied.
    unmatched_overrides = sorted(set(closed_class_overrides or ()) - {fact.subject for fact in declared if fact.predicate == PREDICATE_CLOSED_SET})
    report = build_pre_generation_report(
        verification=verification,
        allowlist=allowlist,
        inventory=inventory,
        probe_set=probe_set,
        smoke_probes=smoke_probes,
        drift=drift,
        corpus_id=corpus_id,
        load_warnings=load_warnings,
        citations=citations,
        governance=governance,
        services=services,
        closed_class_decisions=closed_class_decisions,
        unmatched_closed_class_overrides=unmatched_overrides,
        dropped_facts=dropped_facts,
        empty_roots=empty_roots,
        declaration_problems=declaration_problems,
    )
    report.apply_overrules(overrules)
    return ScanContractResult(
        report=report,
        verification=verification,
        allowlist=allowlist,
        inventory=inventory,
        probe_set=probe_set,
        drift=drift,
        citations=citations,
        governance=governance,
        services=services,
        corpus_id=corpus_id,
        render_surface=render_surface,
        render_surface_unreadable=next((p.detail for p in declaration_problems if p.check == RENDER_ACTIONS_KEY), None) if render_surface is None else None,
        closed_class_decisions=tuple(closed_class_decisions or ()),
        unmatched_closed_class_overrides=tuple(unmatched_overrides),
    )


#: The internal gate record a ``check`` writes BESIDE ``scan_contract.json``: every finding and what
#: each overrule did, overrule reasons included. Never inside the artifact — that one is uploaded,
#: committed to the customer's repository and copied into generation workspaces — and never
#: rendered for the customer.
GATE_AUTHORITY_FILENAME = 'gate_authority.json'


def gate_authority_payload(result: ScanContractResult) -> dict[str, Any]:
    """The sidecar's content (:data:`GATE_AUTHORITY_FILENAME`): the findings and every overrule outcome."""
    return result.report.gate_authority().model_dump(mode='json')


def artifact_payload(result: ScanContractResult) -> dict[str, Any]:
    """The persisted scan-contract artifact: the report plus the fingerprint to re-check later.

    ``source_inventory`` is stored BESIDE ``corpus_id`` deliberately — the id says which corpus
    was built, the inventory says from what. Without the second, "the corpus as of the scan" is
    an assertion nothing can test.

    ``render_actions`` is the third carried input, and it is ABSENT — not null, not empty —
    when no config declared a surface: ``{"verbs": [], ...}`` is "none, by declaration", and a
    consumer must be able to tell that from "nobody said" by the key alone. A declaration that
    exists and could not be read rides as ``{"unreadable": <why>}`` — present, so it reads as
    unusable rather than undeclared. It rides the artifact
    and not the report because the report is what a human reads at scan time and the surface is
    what the backend's ``RENDER_ACTIONS`` check reads at generation time.
    """
    payload = {
        'schema': result.report.schema,
        'corpus_id': result.corpus_id,
        'source_inventory': result.inventory.as_dict(),
        'report': result.report.as_dict(),
        'url_allowlist': result.allowlist.as_dict(),
        'absence_probe_policies': result.probe_set.as_policies(),
        'source_drift': result.drift.as_dict() if result.drift is not None else None,
    }
    if result.render_surface is not None:
        payload[RENDER_ACTIONS_KEY] = result.render_surface.as_record()
    elif result.render_surface_unreadable is not None:
        # PRESENT, as a marker the backend's reader refuses (``RenderSurface.from_record``), so its
        # check blocks as ``unusable`` and not as ``undeclared``.
        payload[RENDER_ACTIONS_KEY] = {RENDER_ACTIONS_UNREADABLE: result.render_surface_unreadable}
    if result.closed_class_decisions:
        # The class RECORDS, not just the verdicts: the backend re-runs the same ladder over its
        # own registry, and a divergence is only diagnosable when both sides publish what they
        # measured. Absent (not empty) when the scan declared no closed set, so "nobody said" and
        # "said, and nothing closed" stay distinguishable by the key alone.
        payload[CLOSED_CLASSES_KEY] = {
            'classes': [d.record for d in result.closed_class_decisions if d.record is not None],
            'unresolved_subjects': [d.fact.subject for d in result.closed_class_decisions if d.record is None],
        }
    return payload


# =====================================================================================
# Reconciling the two corpus snapshots
# =====================================================================================
#
# The scan verifies its facts against the CLIENT-side source directories. Scoring-time
# grounding derives from the SERVER-side ``source_registry.yaml`` + the built corpora the set is
# actually attached to, assembled at a different time from a different selection of bytes.
# The two can differ, and the failure that produces is not a missed check but a
# MANUFACTURED one: an absence true of the client files can be false of the server-side corpus,
# and the probe minted from it carries "the agent must decline" as graded ground truth over a
# mock whose RAG serves the thing. The agent answers correctly and the suite scores it red, with
# a judge prompt that states the corpus does not contain what the corpus contains.
#
# So the comparison happens BEFORE a probe exists, and its output is a pre-generation FINDING —
# never a probe scenario.


@dataclass(frozen=True)
class GroundingDivergence:
    """One contract fact the SERVER-side grounding POSITIVELY contradicts.

    Only the absence kind can produce one, and that is a property of what a second snapshot can
    prove rather than a scoping choice. The server-side grounding is assembled from a different
    selection of bytes, so it is a DIFFERENT SET, not a newer version of the same one — and a
    different set can only ever supply POSITIVE evidence. A hit refutes an absence outright. An
    existence claim the server snapshot cannot satisfy, or a count that comes out lower, says
    nothing at all: those are statements about which bytes reached the appendix and which corpus
    passages were sampled. Reporting them as "the corpus says otherwise" would file a finding
    against every set whose PRD appendix is a subset of the customer's docs, which is all of them.
    """

    fact_id: str
    claim: str
    #: What the same executable check found in the grounding — always a positive hit.
    reason: str
    #: Fixed. Kept as a field rather than implied so the published record says what was compared.
    kind: str = KIND_ABSENCE
    scan_status: str = VERIFIED
    grounding_status: str = CONTRADICTED

    @property
    def blocks_probe(self) -> bool:
        """Always. A probe minted from this fact would assert, as graded ground truth, that a
        thing the agent's own retrieval serves does not exist."""
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            'fact_id': self.fact_id,
            'kind': self.kind,
            'claim': self.claim,
            'scan_status': self.scan_status,
            'grounding_status': self.grounding_status,
            'reason': self.reason,
            'blocks_probe': self.blocks_probe,
        }


@dataclass(frozen=True)
class GroundingReconciliation:
    """Whether the contract's facts still hold against the corpus the SET is grounded on.

    ``checked`` is load-bearing and separate from ``divergences``: an empty divergence list is
    the shape of "compared every fact, they agree" AND of "there was nothing to compare against",
    and only the first licenses minting a probe on the strength of it.
    """

    checked: bool
    #: Why the comparison could not be made. Empty when ``checked``.
    blocked_reason: str = ''
    divergences: tuple[GroundingDivergence, ...] = ()
    #: Fact ids the grounding AGREES with — reported so "0 divergences" has a denominator.
    agreed: tuple[str, ...] = ()
    #: Grounding scopes the comparison ran over. Zero with ``checked`` False.
    scopes: int = 0

    @property
    def blocked_fact_ids(self) -> frozenset[str]:
        """Facts no probe may be minted from — see :attr:`GroundingDivergence.blocks_probe`."""
        return frozenset(d.fact_id for d in self.divergences if d.blocks_probe)

    def findings(self) -> list[str]:
        """One line per divergence, in the pre-generation-finding voice. Never empty when there
        is a divergence, and never non-empty when there is not."""
        return [
            f'{d.fact_id} ({d.kind}): the scan verified this against the CLIENT sources and the corpus this set is '
            f'grounded on CONTAINS it — {d.reason} No probe was minted from it: the probe would have graded the agent '
            'against a premise its own retrieval refutes.'
            for d in self.divergences
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            'checked': self.checked,
            'blocked_reason': self.blocked_reason,
            'scopes': self.scopes,
            'agreed': list(self.agreed),
            'divergences': [d.as_dict() for d in self.divergences],
        }


def reconcile_against_grounding(artifact: Mapping[str, Any] | None, units: Sequence[CorpusUnit]) -> GroundingReconciliation:
    """Re-run the contract's own executable checks against a SECOND corpus.

    ``artifact`` is the published ``scan_contract.json``; the checks come out of its verdicts'
    ``check`` payloads, so this is the SAME predicate the scan ran, not a re-derivation of it —
    a second implementation here would report divergences that are only two matchers disagreeing.

    Only facts the scan VERIFIED are compared. A fact it contradicted or could not decide is
    already a finding upstream, and re-deciding it against a different corpus would
    answer a question nobody asked with a corpus that was never the subject.

    NEVER raises: this runs inside a generation job. Every way it could not look becomes
    ``checked=False`` with the reason, which the caller reports rather than reading as agreement.
    """
    if not isinstance(artifact, Mapping):
        return GroundingReconciliation(checked=False, blocked_reason='no scan contract reached this generation, so there was nothing to reconcile')
    if not units:
        # NOT a failure, and not a pass either. With no server-side grounding there is no
        # retrieval that could serve the absent thing, so no probe can be manufactured — but the
        # comparison did not happen and the ledger has to say so.
        return GroundingReconciliation(
            checked=False,
            blocked_reason='this set is grounded on no server-side source or corpus text, so no fact could be re-decided against it '
            '(nothing the agent retrieves can contradict an absence claim either)',
        )
    verdicts = artifact_fact_verdicts(artifact)
    if verdicts is None:
        return GroundingReconciliation(
            checked=False,
            blocked_reason='the scan artifact publishes no report.sections.fact_verification.verdicts list, so its executable checks could not be re-run',
        )
    verified = [row for row in verdicts if isinstance(row, Mapping) and row.get('status') == VERIFIED]
    if not verified:
        return GroundingReconciliation(checked=True, scopes=len(units))
    try:
        facts, dropped = load_scan_facts(verified)
    except ValueError as exc:
        return GroundingReconciliation(
            checked=False,
            blocked_reason=f"the scan artifact's verified facts could not be read back as checks ({exc}) — it predates "
            'the published `check` payload, or it is malformed',
        )
    if dropped:
        return GroundingReconciliation(
            checked=False,
            blocked_reason=f"the scan artifact's verified facts could not be read back as checks ({dropped[0].reason}) — it "
            'predates the published `check` payload, or it is malformed',
        )
    divergences: list[GroundingDivergence] = []
    agreed: list[str] = []
    for verdict in verify_facts(facts, units).verdicts:
        if verdict.status == VERIFIED:
            agreed.append(verdict.fact.fact_id)
            continue
        # Everything else is DROPPED, and the two reasons are one reason. An unresolvable verdict
        # (a scope glob written against client paths addresses no server unit) is a statement about
        # addressing. A contradicted EXISTENCE or COUNT is a statement about which bytes reached
        # the appendix and which corpus passages were sampled. Neither is evidence about the
        # claim — see :class:`GroundingDivergence` — and filing either would block probes on
        # every set whose server-side snapshot is a subset of the customer's documents.
        if verdict.status == CONTRADICTED and verdict.fact.kind == KIND_ABSENCE:
            divergences.append(GroundingDivergence(fact_id=verdict.fact.fact_id, claim=verdict.fact.claim, reason=verdict.reason))
    return GroundingReconciliation(checked=True, divergences=tuple(divergences), agreed=tuple(agreed), scopes=len(units))


# =====================================================================================
# What a refuted fact takes down with it
# =====================================================================================


@dataclass(frozen=True)
class FactDependents:
    """The suite artifacts built on a set of facts, and the ones whose link cannot be read.

    ``unstampable`` is the half that keeps this honest. A suite built before content-addressed
    fact ids, or one whose facts still carry positional ids, has stamps that name SLOTS rather
    than claims — and answering "nothing depends on this fact" over such a suite is the same
    sentence as answering it over a suite that genuinely has no dependants.
    """

    #: ``{fact_id: [policy name, ...]}`` for the policies stamped ``grounded_on`` that fact.
    policies: dict[str, list[str]] = field(default_factory=dict)
    #: ``{fact_id: [scenario name, ...]}`` from the scenario set's absence-probe census.
    scenarios: dict[str, list[str]] = field(default_factory=dict)
    #: Fact ids the caller asked about that carry no readable dependency link, with why.
    unstampable: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.policies and not self.scenarios

    def render(self) -> str:
        """The rebuild instruction a recheck prints. States the blast radius, not just the map.

        Two things a dependency map cannot say on its own, and both are what makes the rebuild
        actually work:

        * regeneration RENAMES LLM-named flow ids (few of them survive a
          rebuild), so every eval scoped by ``target_flows`` / ``current_elements`` on a
          REBUILT scenario silently unscopes — a set strictly larger than this map. ``relink_eval_scenario_set``
          reports the dead guards; it does not repair them, so they have to be routed into the
          amendment loop by hand.
        * a rebuilt scenario's content hash misses its solvability stamp, so it needs a fresh
          passing run (a partial re-qualification), which is cheap only if it is expected.
        """
        lines: list[str] = []
        for fact_id in sorted({*self.policies, *self.scenarios}):
            targets = [f'scenario {name}' for name in self.scenarios.get(fact_id, ())] + [f'policy {name}' for name in self.policies.get(fact_id, ())]
            lines.append(f'- {fact_id}: {", ".join(targets)}')
        for fact_id, reason in sorted(self.unstampable.items()):
            lines.append(f'- {fact_id}: DEPENDANTS UNKNOWN — {reason}')
        if not lines:
            return (
                'No suite artifact records a dependency on the refuted fact(s). That is a statement about the STAMPS, '
                'not about the suite: an artifact built before provenance stamping records nothing here.'
            )
        return '\n'.join(
            [
                'These suite artifacts stand on the refuted fact(s) and must be REBUILT, not patched:',
                *lines,
                '',
                'After the rebuild, and BEFORE scoring anything:',
                '  1. relink_eval_scenario_set(eval_set_id, <new scenario_set_id>) — regeneration renames LLM-named flow ids, '
                'so every eval scoped by target_flows / current_elements on a rebuilt '
                'scenario unscopes and grades nothing. Relink REPORTS the dead guards; it does not repair them — route each '
                'into /edit-evals.',
                "  2. Expect a PARTIAL RE-QUALIFICATION: a rebuilt scenario's content hash misses its solvability stamp, so it needs a fresh passing run.",
            ]
        )


def fact_dependents(
    refuted_fact_ids: Iterable[str], *, policies: Mapping[str, Any] | None = None, dataset_metadata: Mapping[str, Any] | None = None
) -> FactDependents:
    """Map refuted fact ids to the exact scenarios and policies built on them.

    ``policies`` is a parsed ``policies_config.yaml``; ``dataset_metadata`` is a parsed scenario
    set ``dataset_metadata.yaml``. Either may be absent — the result then simply carries nothing
    from that side, which :meth:`FactDependents.render` states rather than presenting as a clean
    map.

    A refuted fact whose id is POSITIONAL is answered with ``unstampable`` rather than with an
    empty list: nothing may have been stamped with it, because stamping over a slot-name is
    refused upstream, and reporting "no dependants" would tell an operator their suite is safe.
    """
    wanted = [str(f) for f in refuted_fact_ids]
    result = FactDependents()
    for fact_id in wanted:
        if fact_id_is_positional(fact_id):
            result.unstampable[fact_id] = (
                'this is a positional fact id (it names a slot in the facts document, not a claim), so nothing was '
                'stamped with it and the dependency map cannot answer for it. Re-scan with content-addressed fact ids.'
            )
    for name, body in (policies or {}).items():
        stamped = body.get('grounded_on') if isinstance(body, Mapping) else None
        for fact_id in stamped if isinstance(stamped, list) else []:
            if str(fact_id) in wanted:
                result.policies.setdefault(str(fact_id), []).append(str(name))
    probes = (dataset_metadata or {}).get('absence_probes')
    census = probes.get('dealt') if isinstance(probes, Mapping) else None
    grounded = census.get('grounded_on') if isinstance(census, Mapping) else None
    for fact_id, names in grounded.items() if isinstance(grounded, Mapping) else ():
        if str(fact_id) in wanted and isinstance(names, list):
            result.scenarios.setdefault(str(fact_id), []).extend(str(n) for n in names)
    return result
