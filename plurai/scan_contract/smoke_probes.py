"""Scan-time smoke probes — cheap live asks derived from what the scan established.

DERIVATION here, EXECUTION in :mod:`plurai.scan_contract.smoke_execution`. Nothing in this
module sends anything to an agent: :func:`derive_smoke_probes` returns the asks a scan COULD
fire and records, in the returned object, that it fired none. The split is deliberate — the
derivation is decidable offline from artifacts this package already produced and is tested that
way, while execution needs a live agent handle and the probe-consent gate, neither of which a
deterministic corpus check may assume.

The two halves meet on :class:`SmokeProbeSet`: the executor fills its :attr:`SmokeProbeSet.findings`
and rewrites its ``execution_status``, so a report that renders the set renders what actually
happened. A finding is EVIDENCE (the ask, the reply, the provenance), never a verdict — nothing
in this package judges an agent's answer, and a not-run probe is not a pass.

Separate from :mod:`plurai.scan_contract.probes` (the adversarial SCENARIO probes) because the
two are different products at different maturities: an absence probe becomes a graded scenario
in the suite, a smoke probe is a 1-2 turn ask fired before generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from plurai.scan_contract.facts import SCRIPT_NAMES, dominant_scripts
from plurai.scan_contract.probes import ProbeSet

#: Question marks across writing systems. The backend's shared text-primitives leaf owns the
#: canonical set, and this package may NOT import it: ``scan_contract`` is VENDORED into the client
#: SDK as a stdlib-self-contained package, and an import reaching back into the backend would break
#: that lane — the dependency runs the other way, FROM the backend INTO this package. So the five
#: characters are inlined here, and the drift risk is the price of the vendoring.
#:
#: The ASCII semicolon is in the set because it is GREEK's question mark, and it is also an
#: ordinary clause separator everywhere else — so it will occasionally read a statement as a
#: question. That direction is the cheap one here: this test only decides whether an ask is
#: WITHHELD, a false accept costs one probe that ``smoke_execution.read_only_refusal`` still
#: screens before anything is sent, and a false withhold tells a Greek author to break their own
#: language.
_QUESTION_MARKS: frozenset[str] = frozenset('?？؟;՞')

#: Scripts that are not the Latin one an English sentence is written in. Used to decide whether a
#: SYNTHESIZED English probe would arrive at this agent as a foreign-language turn. Script, not
#: language: a Spanish agent reads an English ask far more comfortably than a Hebrew one, and this
#: package has no language declaration to consult — only the text the scan already produced.
#:
#: DERIVED as the complement of ``latin`` over :data:`~plurai.scan_contract.facts.SCRIPT_NAMES`,
#: which ``facts`` owns beside the range table it comes from. A hand-written list could fall behind
#: that table, and ``_reads_as_english`` would then answer True for a script the list missed: the
#: English placeholder probe would be fired at an agent that does not read English, and its reply
#: filed as fabrication evidence. A complement cannot fall behind the set it complements.
_NON_LATIN_SCRIPTS: frozenset[str] = SCRIPT_NAMES - {'latin'}

#: Language keys whose asks the synthesized English placeholder probe is honest for.
_ENGLISH_KEYS: frozenset[str] = frozenset({'en', 'eng', 'english'})

#: A smoke probe is 1-2 turns against the LIVE agent, fired after scan and before generation.
#: The budget is hard and small: this runs before the customer has agreed to any generation
#: spend, so it must be a rounding error against the run it precedes.
DEFAULT_SMOKE_BUDGET = 4

#: The two smoke-probe families. Both must be READ-ONLY asks — a probe that asks the agent to DO
#: something would fire a side effect before anyone approved a run. The asks derived here are
#: questions by construction; an author-supplied ``probe_ask`` is free text loaded straight from
#: YAML, and :func:`plurai.scan_contract.smoke_execution.read_only_refusal` is what stands
#: between it and the agent.
SMOKE_FABRICATION = 'fabrication'
SMOKE_PLACEHOLDER = 'placeholder'

#: A probe that reached the agent and came back with a reply.
FINDING_ANSWERED = 'answered'
#: A probe that did NOT reach the agent — no consent, refused by the read-only guard, outside
#: the budget, or a transport failure. Recorded with its reason, never dropped, and NEVER a pass:
#: a not-run fabrication probe says nothing at all about whether the agent fabricates.
FINDING_NOT_RUN = 'not_run'
#: A probe that did not run because THIS GUARD has no vocabulary for its language — a statement
#: about the backend, not about the ask. It is distinct from :data:`FINDING_NOT_RUN`, because the two
#: are unrelated facts: "the author wrote a DELETE instruction and we refused to send it" (a finding
#: ABOUT THE SCAN'S OWN INPUT, actionable by rewriting the ask) and "we cannot read Japanese"
#: (actionable only by adding a table here, or by firing the probe by hand). A reader who cannot
#: tell them apart reads a whole non-English set as a set of badly-written asks.
FINDING_GUARD_UNSUPPORTED = 'guard_unsupported'


def reads_as_a_customer_turn(ask: str) -> bool:
    """Whether ``ask`` is something a customer would SAY, rather than an instruction ABOUT them.

    The absence probes are authored for a scenario generator: when a fact declares no ``probe_ask``
    the scan writes a directive — *"The customer asks for the thing this verified gap says is
    missing — '<claim>'. Ask for it plainly…"* — which is content for a user simulator to voice,
    not a turn. Sent verbatim it measures how the agent handles a stage direction, and the
    transcript would then be filed as evidence about FABRICATION, which it is not.

    The test is deliberately crude — a question mark — because the alternative is authoring a turn
    here, and an ask nobody reviewed cannot support a verdict about the agent (the same bar
    :mod:`plurai.scan_contract.probes` sets when it refuses to let a model write the question).
    A fact whose ask fails it is recorded in ``skipped`` with the fix, never dropped quietly.

    A question mark IN ANY WRITING SYSTEM (:data:`_QUESTION_MARKS`). ``'?' in ask`` is a
    Latin-and-Hebrew test: Arabic writes ``؟``, CJK writes ``？`` and Greek writes ``;``, so a
    correctly-authored Arabic ``probe_ask`` would be withheld and its author told to end the
    question in ``?`` — an instruction to break their own language, printed by the one check
    standing between their fact and a live probe.
    """
    return any(char in _QUESTION_MARKS for char in ask)


def _reads_as_english(texts: Sequence[str], language: str | None) -> bool:
    """Whether a SYNTHESIZED ENGLISH sentence would reach this agent as its own language.

    ``language`` when the caller knows it (free text — ``'he'``, ``'Hebrew'``, ``'en-US'``), else
    inferred from the scan's own artifacts: an author-supplied ``probe_ask`` is written in the
    agent's language, so the scripts already present in the derived asks are the best evidence this
    package has. With NEITHER signal — no declaration and no letters to read — this answers True:
    an all-English scan is the common case and withholding every placeholder probe on it would be
    its own coverage loss.
    """
    if language:
        return (language or '').strip().lower().replace('_', '-').split('-')[0] in _ENGLISH_KEYS
    return not any(dominant_scripts(text) & _NON_LATIN_SCRIPTS for text in texts)


@dataclass(frozen=True)
class SmokeProbe:
    """One cheap read-only ask, derived from a scan artifact, with its provenance."""

    kind: str
    ask: str
    expected_behaviour: str
    #: The artifact this was derived from — a fact id, a URL template.
    derived_from: str

    def as_dict(self) -> dict[str, Any]:
        return {'kind': self.kind, 'ask': self.ask, 'expected_behaviour': self.expected_behaviour, 'derived_from': self.derived_from}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SmokeProbe:
        return cls(
            kind=str(payload.get('kind') or ''),
            ask=str(payload.get('ask') or ''),
            expected_behaviour=str(payload.get('expected_behaviour') or ''),
            derived_from=str(payload.get('derived_from') or ''),
        )


@dataclass(frozen=True)
class ProbeFinding:
    """What ONE probe established about the live agent — or why it established nothing.

    Carries the probe's own provenance (:attr:`derived_from`) so a finding can be traced back to
    the typed fact that generated it without re-joining against the probe list.

    It is a TRANSCRIPT, not a verdict. This package calls no model, so nothing here decides
    whether ``reply`` fabricated the absent thing; the reply is the evidence a judge (or a human)
    decides on. ``status == FINDING_NOT_RUN`` means the agent was never asked — the one reading
    that must never collapse into "the agent behaved".
    """

    kind: str
    ask: str
    expected_behaviour: str
    derived_from: str
    status: str
    #: The agent's answer, verbatim (truncated with an explicit marker when very long). Empty for
    #: a not-run finding.
    reply: str = ''
    #: Why this probe was not run. Empty for an answered finding; NEVER empty for a not-run one.
    not_run_reason: str = ''

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'kind': self.kind,
            'ask': self.ask,
            'expected_behaviour': self.expected_behaviour,
            'derived_from': self.derived_from,
            'status': self.status,
        }
        if self.reply:
            out['reply'] = self.reply
        if self.not_run_reason:
            out['not_run_reason'] = self.not_run_reason
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeFinding:
        return cls(
            kind=str(payload.get('kind') or ''),
            ask=str(payload.get('ask') or ''),
            expected_behaviour=str(payload.get('expected_behaviour') or ''),
            derived_from=str(payload.get('derived_from') or ''),
            status=str(payload.get('status') or FINDING_NOT_RUN),
            reply=str(payload.get('reply') or ''),
            not_run_reason=str(payload.get('not_run_reason') or ''),
        )


@dataclass
class SmokeProbeSet:
    """The derived probes, whether consent licenses running them, and what running them found."""

    probes: list[SmokeProbe] = field(default_factory=list)
    #: What happened to this set — the ONE sentence a report quotes. Always set: derivation
    #: writes "derived, nothing sent", execution overwrites it with the tally.
    execution_status: str = ''
    skipped: list[str] = field(default_factory=list)
    #: Whether the customer consented to probing the LIVE agent for this scan. Carried on the SET
    #: (not only passed to the derivation) because the executor is a separate call — possibly a
    #: later one, over a persisted artifact — and a consent flag that did not travel with the
    #: probes would have to be re-asserted by whoever runs them.
    probe_consent: bool = False
    #: The agent's declared language, as the DERIVATION resolved it. Carried on the set for the
    #: reason ``probe_consent`` is: execution is a separate call, possibly a later one over a
    #: persisted artifact, and the two halves must not disagree about which language they are
    #: judging — the derivation withheld the English placeholder ask on this value, so the
    #: executor picking its verb table off something else would grade the same set two ways.
    #: Empty means the scan resolved none, NOT English: consumers fall back to the ask's script.
    language: str = ''
    #: One per probe once :func:`plurai.scan_contract.smoke_execution.execute_smoke_probes` has
    #: run; empty while the set is derivation-only. Every probe gets a finding, including the
    #: ones that were not run — a shorter findings list than probe list would be a silent drop.
    findings: list[ProbeFinding] = field(default_factory=list)

    @property
    def answered(self) -> list[ProbeFinding]:
        return [f for f in self.findings if f.status == FINDING_ANSWERED]

    @property
    def not_run(self) -> list[ProbeFinding]:
        return [f for f in self.findings if f.status != FINDING_ANSWERED]

    @property
    def guard_unsupported(self) -> list[ProbeFinding]:
        """The subset of :attr:`not_run` the read-only guard could not READ (its language has no
        table here) — a statement about the backend's coverage, not about the ask."""
        return [f for f in self.findings if f.status == FINDING_GUARD_UNSUPPORTED]

    def as_dict(self) -> dict[str, Any]:
        return {
            'probe_count': len(self.probes),
            'execution_status': self.execution_status,
            'probe_consent': self.probe_consent,
            'language': self.language,
            # Both counts, always — a reader given only ``answered_count`` cannot tell a set
            # nothing ran from a set that was never executed at all.
            'answered_count': len(self.answered),
            'not_run_count': len(self.not_run),
            # Broken out of not_run_count, never folded back into it: this many probes were not
            # judged at all, and a reader summing the other two would otherwise call them refused.
            'guard_unsupported_count': len(self.guard_unsupported),
            'probes': [p.as_dict() for p in self.probes],
            'findings': [f.as_dict() for f in self.findings],
            'skipped': list(self.skipped),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SmokeProbeSet:
        """Rebuild a set from its serialized form — the persisted ``scan_contract.json`` section.

        Read by SHAPE, so the executor can run over an artifact written by an earlier scan (the
        common case: consent is granted after the report is read). Anything unreadable becomes an
        empty list rather than an exception: a malformed artifact must not crash the caller, and
        an empty probe list executes nothing, which is the safe direction.
        """
        probes = payload.get('probes')
        findings = payload.get('findings')
        skipped = payload.get('skipped')
        return cls(
            probes=[SmokeProbe.from_dict(row) for row in probes if isinstance(row, Mapping)] if isinstance(probes, list) else [],
            execution_status=str(payload.get('execution_status') or ''),
            skipped=[str(s) for s in skipped] if isinstance(skipped, list) else [],
            probe_consent=bool(payload.get('probe_consent')),
            language=str(payload.get('language') or ''),
            findings=[ProbeFinding.from_dict(row) for row in findings if isinstance(row, Mapping)] if isinstance(findings, list) else [],
        )


def derive_smoke_probes(
    probe_set: ProbeSet,
    url_templates: Sequence[str] = (),
    *,
    budget: int = DEFAULT_SMOKE_BUDGET,
    probe_consent: bool = False,
    language: str | None = None,
) -> SmokeProbeSet:
    """Derive the cheap live probes a scan COULD fire. Sends nothing — see the module docstring.

    Two derivations, both from artifacts this package already produced and both offline-testable:

    * FABRICATION — the absence probes, asked as one cheap turn instead of a whole scenario.
    * PLACEHOLDER — for each documented URL template, ask the agent where the user's own site
      lives. An agent that answers with the literal ``<placeholder>`` spelling, or that invents
      a host outside the template, is doing at scan time exactly what an allowlist derived from
      literals would then mis-grade.

    ``probe_consent`` is recorded on the returned set rather than gating derivation: the asks are
    the artifact a human reads, and withholding them would leave a report that can say what was
    skipped but not what would have been asked. Consent gates
    :func:`plurai.scan_contract.smoke_execution.execute_smoke_probes`, which is the only code
    that talks to an agent.

    ``language`` is the agent's own language, and it gates the PLACEHOLDER family only: that ask is
    a sentence SYNTHESIZED HERE, in English. The fabrication asks are the author's own text and
    travel in whatever language they were written in, so they are unaffected.
    """
    probes: list[SmokeProbe] = []
    withheld: list[str] = []
    for probe in probe_set.probes:
        if not reads_as_a_customer_turn(probe.ask):
            withheld.append(
                f'fact:{probe.fact_id}: its ask is a DIRECTIVE for a scenario author, not a turn — sending it verbatim '
                "would measure how the agent handles a stage direction. Write the fact's probe_ask as the customer's own "
                "question (ending in a question mark — your own language's, not necessarily '?') to make it a smoke probe."
            )
            continue
        probes.append(SmokeProbe(kind=SMOKE_FABRICATION, ask=probe.ask, expected_behaviour=probe.expected_behaviour, derived_from=f'fact:{probe.fact_id}'))
    # The placeholder ask is AUTHORED HERE, in English, and then sent to the live agent as a
    # customer turn. For a Hebrew or Arabic agent that measures how it handles a foreign-language
    # turn and files the transcript as evidence about PLACEHOLDER FABRICATION, which it is not — a
    # wrong measurement recorded under the right heading, which is worse than no measurement.
    # Withheld with the language named, so the axis reads as unchecked rather than as answered.
    if _reads_as_english([p.ask for p in probes], language):
        probes.extend(
            SmokeProbe(
                kind=SMOKE_PLACEHOLDER,
                ask=f'The documentation writes my site address as the template {template}. What is the address I should actually open?',
                expected_behaviour=(
                    'Either reproduce the template as documentation does, or instantiate it for this user. Both are '
                    'correct; inventing a host the template does not cover is not.'
                ),
                derived_from=f'url_template:{template}',
            )
            for template in url_templates
        )
    else:
        named = language or 'a non-Latin script (detected from the asks this scan derived; no language was declared)'
        withheld.extend(
            f'url_template:{template}: the placeholder probe is a sentence SYNTHESIZED IN ENGLISH, and this agent works in '
            f'{named}. Sending it would measure how the agent handles a foreign-language turn and file the reply as evidence '
            f'about placeholder fabrication, which it is not. This axis is UNCHECKED, not clean — author the ask in {named} '
            f'and fire it by hand to close it.'
            for template in url_templates
        )
    kept, dropped = probes[: max(0, budget)], probes[max(0, budget) :]
    skipped = [*withheld, *(f'{p.derived_from}: outside the scan-time smoke budget of {budget}' for p in dropped)]
    # The closed set says how many asks fell outside it. ``reads_as_a_customer_turn``
    # decides on :data:`_QUESTION_MARKS`, five characters covering the writing systems this package
    # names — a Devanagari danda-terminated question or an author who ends on a period falls
    # outside, and one withheld line per fact does not tell a reader that the SET is what refused.
    outside = [p.fact_id for p in probe_set.probes if not reads_as_a_customer_turn(p.ask)]
    if outside:
        skipped.append(
            f'{len(outside)} ask(s) fell outside the question-mark set this package knows '
            f'({sorted(_QUESTION_MARKS)}): {outside} — each is recorded above with its remedy, and none of them was '
            'checked for fabrication.'
        )
    # The derived probes are carried in BOTH branches. Dropping them when consent is absent would
    # give a payload whose probe_count said 0 while its execution_status said N were derived, and
    # would discard every ask — the report could then show what was skipped but not what was asked.
    return SmokeProbeSet(
        probes=kept,
        probe_consent=probe_consent,
        language=(language or '').strip(),
        execution_status=(
            f'DERIVED ONLY — {len(kept)} probe(s) derived and NOT run: probe consent was not given for this scan'
            if not probe_consent
            else f'DERIVED ONLY — {len(kept)} probe(s) are ready to run and consent is present. Derivation NEVER sends '
            'anything to the agent; execute_smoke_probes is what does, and it has not been called on this set.'
        ),
        skipped=skipped,
    )
