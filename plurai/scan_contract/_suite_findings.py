"""The typed finding schema `suite_diagnostics` emits — vocabulary, ids, and check accounting.

Three properties carry the whole design, and each is here rather than in the checks because
each is a property of the REPORT, not of any one check:

**Ids are content-addressed on class + normalized locus, and on nothing else.** No run hash,
no scenario-set id, no timestamp. The point of an id is that the NEXT run's report can say
``verified-fixed`` / ``still-present`` about a finding this run reported, and a run-scoped id
can never match across runs — it would make the regression contract structurally impossible
while looking implemented.

**Every check is three-valued.** ``ran`` (looked, found nothing), ``flagged`` (looked, found
something) and ``blocked`` (could not look — the input was missing) are three different
facts, and a report that collapses the third into the first is the coverage-theater shape
this whole effort exists to remove: a suite where every check silently no-opped renders
byte-identically to a clean one.

**The vocabulary is closed and versioned with the schema.** Consumers switch on ``class`` and
``owning_layer``; an unknown value must be READ as unknown rather than dropped, so the
schema states the open-enum rule explicitly for older plugin versions. Every member below
MUST have a live emitter: a class a reader can see in the vocabulary and never in a report
is a coverage claim the engine does not keep, which is the same false green the report
exists to remove.

Deliberately ABSENT: ``repair_channel``. Which gated tool repairs a finding is the flow's
knowledge, and a closed vocabulary shared between the backend and the plugin, with no check
comparing the two sides, drifts apart. The backend says WHICH LAYER OWNS the defect;
the plugin maps that to a channel.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

#: Bumped whenever the closed vocabularies below change meaning. A consumer reading a
#: schema it does not know must degrade to rendering the findings verbatim, never to
#: dropping the ones whose class it cannot resolve.
SCHEMA_VERSION = 'suite-diagnostics-v17'

#: WHAT is wrong. Named after the EXISTING typed outputs wherever one exists (``not_scored``
#: reasons, guard-skip kinds, linked-set audit findings) rather than inventing a parallel name for
#: the same fact.
FindingClass = Literal[
    'policy_never_scored',  # a registered policy produced no verdict anywhere in the run
    'policy_underexposed',  # it DID score somewhere, on too little of the run to read as coverage
    'co_exclusive_pair',  # two policies where exactly one failed on every unit both graded — the agent could satisfy only one
    'guard_raises',  # an applicability_guard errored — the policy is mute wherever it evaluates
    'inverted_policy_shipped',  # the benign-run probe says the polarity is flipped, and the run agrees
    'unevidenced_violation',  # a boolean judge claimed a violation and cited no instance of it
    'double_charge',  # two policies charged ONE slip — the same (turn, quote) — with no declared deferral between them
    'gt_expression_error',  # the scenario's own deterministic check raised instead of deciding
    'gt_satisfiable_at_turn_zero',  # the deterministic check already passes before the agent acts
    'gt_criterion_unverified',  # a declared success criterion produced no verdict at all
    'gt_check_disputed',  # the deterministic goal check failed on every trial while the target judge passed
    'currency_mismatch',  # the agent stated a money fact's amount in a currency that is not the fact's
    'unforced_premise_fact',  # a fact the scenario planted for a tool never appears in the tool trail
    'premise_withheld_by_mock',  # that fact is seeded, and the mock answered without it (not found, or the passage left out)
    'premise_stripped_by_product',  # the agent opened the record holding that fact, and the product's reader removed it
    'mock_answer_unverified',  # a runtime tool's served answer states figures no quote it cited spells out
    'mock_answer_refused',  # a runtime tool's answer was refused after its repair and served as not_found
    'unforced_premise_complication',  # a complication the draw dealt, and a judge is armed by, the run never triggered
    'measurement_integrity_breach',  # a verdict-affecting error the headline numbers do not reflect
    'user_sim_fidelity_violation',  # the simulator departed from the schedule the scenario declares
    'user_sim_run_invalid',  # the simulated user repeated itself — the run measures the simulator, not the agent
    'user_sim_repeat_recovered',  # the simulator repeated itself and was re-drawn — the run is valid, it is not
    'user_sim_validity_unknown',  # the run-validity check could not run — unknown, which is not clean
    'user_sim_fidelity_unchecked',  # the evaluator-voice judge did not answer for the run's turns — not measured, which is not clean
    'user_sim_self_report_inconsistent',  # the simulator's own delivery ledger disagreed with itself; the transcript may be on schedule
    'served_guard_drift',  # one policy's guard text differs across scenarios of ONE run
    'scenario_run_failed',  # the scenario errored during the run and carries no transcript to score
    'scenario_unsolvable',  # the scenario's known-good reference path could not pass its own eval set
    'eval_unsatisfiable',  # one policy blocked every measured reference path and was satisfied by none
    # the harness owns the world this suite scores and exports no reset(), so every scenario after the first ran on its predecessor's writes.
    # Only the client runner can detect it, and it refuses at run start, so the backend never emits it; the class is in this vocabulary so a
    # report that names it survives the schema (``owning_layer: harness@client``, ``audience: suite``).
    'world_owner_unreset',
]

#: WHO owns the defect. The backend's half of the repair contract; the plugin maps a layer to
#: one of its gated channels (see the module docstring for why the mapping is not here).
OwningLayer = Literal[
    'element@variables',
    'policy@eval_set',
    'gt@scenario',
    'facts@scenario',
    'harness@client',
    'generator@backend',
    'product_truth@prd',
]

#: WHO the finding is for. ``needs_external_truth`` is the third value on purpose: it gates
#: ``edit_scenario_gt``, which must never be driven from a finding the backend derived on its own.
Audience = Literal['suite', 'customer', 'needs_external_truth']

#: HOW the finding was established. Precedence for a consumer resolving two findings about
#: one locus: deterministic > heuristic > llm_high > llm_low. ``heuristic`` is a rule over recorded
#: facts that has a named way of being wrong, stated in the finding's caveats.
Confidence = Literal['deterministic', 'heuristic', 'llm_high', 'llm_low']

#: WHAT would confirm the fix. ``score_run`` is cheap (judge tokens only), ``rerun`` needs a
#: scored run, ``next_generation`` cannot be confirmed until the artifact is regenerated.
VerifiableVia = Literal['score_run', 'rerun', 'next_generation']

#: HOW WIDE the defect is. A ``set``-scoped finding is not fixed by editing one slot.
Scope = Literal['slot', 'policy', 'set', 'global']

Severity = Literal['critical', 'high', 'medium', 'low']

#: Three-valued, never two. See the module docstring.
CheckStatus = Literal['ran', 'flagged', 'blocked']

#: The verdict tokens the coverage matrix uses. Deliberately few and mutually exclusive: the
#: matrix answers "did this policy produce a verdict on this scenario, and what kind", and a
#: richer vocabulary here would tempt a reader to use the matrix as the verdict record.
#: They live in the SCHEMA module rather than beside the matrix builder because the fault
#: attribution reads the same matrix — two spellings of 'violated' in one report is how a
#: histogram silently stops counting a whole class.
VERDICT_OK = 'pass'  # noqa: S105 - a verdict token, not a credential
VERDICT_VIOLATED = 'violated'
VERDICT_INCONCLUSIVE = 'inconclusive'
VERDICT_NOT_APPLICABLE = 'not_applicable'
VERDICT_NOT_MEASURED = 'not_measured'
#: A verdict this judge REACHED and a named sibling owns.
#: Its own token for the reason ``not_measured`` has one: without it a deferral resolves to
#: ``inconclusive``, and ``run_attribution`` files every ``inconclusive`` cell as an
#: ``eval_defect`` — "the judge was asked a question it could not decide" — over a cell whose
#: judge decided and was deliberately voided by a ``defers_to`` the author wrote on purpose.
VERDICT_DEFERRED = 'deferred'
#: A verdict this judge REACHED and the run's own second sample of the scenario REVERSED.
#: Its own token for exactly the reason ``deferred`` has one: read as ``inconclusive`` it reaches ``run_attribution`` as an
#: ``eval_defect / undetermined`` at ``deterministic`` confidence — "asked a question it could
#: not decide" — over a cell that was decided on both attempts, in opposite directions. Which
#: of the two samples was right is exactly what the run cannot say, and that is the fact the
#: token carries; it charges neither the judge nor the agent.
VERDICT_UNREPRODUCED = 'unreproduced'
VERDICT_ERROR = 'error'
VERDICT_NOT_SCORED = 'not_scored'
VERDICT_REFUSED = 'refused'
VERDICT_ABSENT = 'absent'
#: A cell the run RECORDED under an earlier revision of its policy (its ``policy_fingerprint`` is not the
#: current one): a verdict about a policy that no longer exists, so no finding and no coverage figure reads it.
VERDICT_STALE_REVISION = 'stale_revision'
#: A verdict this judge attempted K times and could not converge on: the guard ARMED (the
#: run-state precondition was true) and the judge WAS invoked; it simply disagreed with itself.
#: Its own token for the same reason ``deferred``/``unreproduced`` needed one: read as plain ``inconclusive`` it is
#: indistinguishable from a single-shot judge that never trusted its own one answer, and read as
#: ``not_scored`` it would silently grace an unbounded amount of judge non-convergence.
VERDICT_LOW_AGREEMENT = 'low_agreement'
#: The OTHER polarity of an agreement-gated cell: the judge did not converge with itself and
#: a STRICT majority of its votable samples said the agent violated (a tie stays
#: :data:`VERDICT_LOW_AGREEMENT`). Its own token because the two polarities are different facts for
#: the release gate: a lean-pass or tied hedge on a scenario another trial decided costs the suite
#: nothing, while a contested violation is a likely defect the gate NAMES as a risk and leaves out of its floor — never a block.
VERDICT_CONTESTED_VIOLATION = 'contested_violation'

#: Everything that is NOT a Unicode letter or digit, plus ``_``, collapsed to one ``-``.
#: Unicode-wide rather than ``[^a-z0-9]``: a locus carries an LLM-authored ``scenario_facts``
#: key or a scenario folder name, neither of which is constrained to Latin, and an
#: ASCII-only class erases a non-Latin locus ENTIRELY — two differently-named Hebrew slots
#: then share one id and ``_dedupe`` merges two unrelated findings into one.
_ID_NORMALIZE = re.compile(r'[\W_]+')


def normalize_locus(locus: str) -> str:
    """The locus half of a finding id, in a form two runs can agree on.

    Lowercased, non-alphanumerics collapsed to a single ``-``. A regenerated slot keeps its
    folder name (``scenario_03``), which is what makes an id survive regeneration — and is
    also why a locus should name the SLOT and the POLICY rather than anything that carries a
    generation's identity.
    """
    return _ID_NORMALIZE.sub('-', locus.strip().lower()).strip('-')


def finding_id(finding_class: str, locus: str) -> str:
    """``sdx-<12 hex>`` over ``class + normalized locus`` — and over nothing else.

    NOT over the run, the upload, the scenario set or the time. Those all make an id unique,
    which is the opposite of what it is for: two runs must produce the SAME id for the same
    defect at the same locus, or ``prior_findings`` can never resolve to anything but
    "everything is new".

    **SO AN ID IS UNIQUE WITHIN A SUITE AND NOT ACROSS SUITES, and a consumer must never treat
    it as globally unique.** That is the price of the paragraph above, paid deliberately, and it
    is visible in the reports: a slot-scoped locus opens with the scenario's own folder name
    (``scenario_03_refund_request/user_sim/turn_2``), so two
    customers' slot findings can never collide — but a policy-scoped one is a bare
    ``policy/<name>``, and two unrelated customers who both register a policy called
    ``knowledge_accuracy`` get one id between them (the same id can appear in two suites' reports).

    **A consumer that resolves ``prior_findings`` owes it a SUITE SCOPE of its own**: read
    this suite's prior report, never a store shared across suites. The scope does not come from
    folding a set id in here. ``scenario_set_id`` is a generation's identity and ``eval_set_id``
    moves on an additive extend, so either one mints a fresh id on every rebuild and destroys exactly the
    stability this function exists for. A cross-suite consumer would need something durable per
    CUSTOMER to scope by; the backend carries no such key.
    """
    digest = hashlib.sha256(f'{finding_class}\x00{normalize_locus(locus)}'.encode()).hexdigest()[:12]
    return f'sdx-{digest}'


@dataclass(frozen=True)
class Evidence:
    """A pointer INTO the bundle, not a re-derivation of it.

    ``artifact`` names the file (``transcript.json``, ``tool_events.json``,
    ``policies_config.yaml``, ``scenario.yaml``), ``path`` the location within it. ``excerpt``
    is optional and deliberately short: evidence payloads can embed real customer records, and
    a report is retained alongside the run upload it was derived from.
    """

    artifact: str
    path: str
    excerpt: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'artifact': self.artifact, 'path': self.path}
        if self.excerpt is not None:
            out['excerpt'] = self.excerpt
        return out


#: Excerpts are truncated here rather than at each call site, so no check can decide to
#: include more of a customer's record than the schema allows. PUBLIC: a second report type
#: (the pre-generation scan findings) quotes customer records under the same schema, and two
#: independent caps is one cap that does not move when the other does.
EXCERPT_CHARS = 240


@dataclass(frozen=True)
class Finding:
    """One typed, actionable defect. Actionable means: no re-derivation needed to act on it."""

    finding_class: FindingClass
    #: What the id is computed over. Name the durable thing — a slot folder, a policy name —
    #: never a generation-scoped identifier.
    locus: str
    severity: Severity
    audience: Audience
    owning_layer: OwningLayer
    scope: Scope
    confidence: Confidence
    verifiable_via: VerifiableVia
    summary: str
    suggested_action: str
    evidence: tuple[Evidence, ...] = ()
    #: Trust limits stamped by the check itself — e.g. that a guard verdict was decided
    #: against the STORED eval-set YAML, which the serving-drift finding shows can differ
    #: from what was served. A caveat is part of the finding, not a footnote to the report.
    caveats: tuple[str, ...] = ()
    #: Which scenarios this finding was observed on. Empty for a set-level finding.
    scenarios: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return finding_id(self.finding_class, self.locus)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'id': self.id,
            'class': self.finding_class,
            'locus': self.locus,
            'severity': self.severity,
            'audience': self.audience,
            'owning_layer': self.owning_layer,
            'scope': self.scope,
            'confidence': self.confidence,
            'verifiable_via': self.verifiable_via,
            'summary': self.summary,
            'suggested_action': self.suggested_action,
            'evidence': [e.as_dict() for e in self.evidence],
        }
        if self.caveats:
            out['caveats'] = list(self.caveats)
        if self.scenarios:
            out['scenarios'] = list(self.scenarios)
        return out


def excerpt(text: object, *, limit: int = EXCERPT_CHARS) -> str:
    """One evidence excerpt, whitespace-collapsed and capped.

    ``limit`` exists so a second caller with its own schema cap goes through THIS function rather
    than writing the algorithm again: two independent truncations of the same judge free text are
    two places that can each decide to keep more of a customer record than a schema allows."""
    flat = ' '.join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + ' …'


@dataclass
class CheckLedger:
    """What each check DID, so silence is attributable.

    A check must call exactly one of :meth:`ran`, :meth:`flagged` or :meth:`blocked` — the
    ledger is the only reason a reader can tell "the premise check looked at every scenario
    and found nothing" from "the premise check had no fact ledger to read".
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, name: str, status: CheckStatus, detail: str = '', *, finding_count: int = 0) -> None:
        self.entries[name] = {'status': status, 'detail': detail, 'finding_count': finding_count}

    def ran(self, name: str, detail: str = '') -> None:
        self.record(name, 'ran', detail)

    def flagged(self, name: str, count: int, detail: str = '') -> None:
        self.record(name, 'flagged', detail, finding_count=count)

    def blocked(self, name: str, detail: str) -> None:
        """The input this check needs was absent. ``detail`` MUST say which one — a blocked
        check with no reason is the same dead end as no check at all."""
        self.record(name, 'blocked', detail)

    def as_dict(self) -> dict[str, Any]:
        return dict(sorted(self.entries.items()))
