"""Closed-fact-class DETECTORS: derive the rule from the members, prove it, refuse when it cannot.

A closed fact class says "the customer's sources enumerate this completely", which licenses a
consumer to treat a value outside the list as one the sources never declared. That licence is
powerful, so a class needs code behind it, and that code cannot be keyed by the class NAME: a
name-keyed detector leaves every other class a sentence in a judge's prompt with nothing checking
it, and the comparison those classes would inherit is URL semantics (scheme / ``www.`` /
trailing-slash stripping, the query asymmetry), which is wrong for a SKU.

So the detector is derived from the MEMBERS, never from the name, by a four-rung ladder:

* **shape** — a closed registry of pre-validated forms (URL, email, phone, UUID, file path,
  handle, version, currency amount, date, numeric id), selected by testing the members against
  each predicate. The registry is a fixed enumeration; a customer pattern is
  rung 2. The cost of that honesty is that ISO country codes and ``KS-12/A`` room codes fall
  through, which is the correct place for them.
* **induced** — the members share a skeleton (``AB-1234``), induced from the members themselves
  over character classes and lengths, never over a dictionary.
* **near_miss** — an IDENTIFIER class with no liftable shape (a ticket ref the shape registry
  does not carry). It never asks "which token is an identifier" — the question that cannot be
  answered without inventing the thing being policed. It lifts only what is a member or one edit
  from one: the corruptions, plus the exact members for the exoneration side.
* **none** — no detector, stated as NOT CHECKED IN CODE with the reason, so a dark class stops
  looking like a clean one.

A rung ACTIVATES only when all SIX gates hold (:func:`derive_detector`). A missing check costs a
miss; a bad check invents a failure and destroys trust in the whole report, so the bias is hard
toward refusing, and ``not_measured`` is a distinct value from ``passed`` everywhere it can arise.

WHY THIS MODULE IS HERE and not beside its main consumer, the backend's source-fidelity judge
support: this package is VENDORED into the client SDK, so the same gates run client-side over the
scanned corpus roots and backend-side over the registered units — one implementation, two
haystacks. The consumer's own package pulls the whole judge/LLM closure the SDK must not acquire.

ONE URL SPELLING. :func:`url_licensed` lives here and is the URL rung's equivalence — the callable
itself, not a copy. Every other URL-membership check in the backend calls it. Two spellings of
the rule would let a code_eval license what a judge reds.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from plurai.scan_contract._dialog_urls import extract_urls, is_url_template, url_matches
from plurai.scan_contract.facts import (
    DETECTOR_RUNGS,
    RUNG_INDUCED,
    RUNG_NEAR_MISS,
    RUNG_NONE,
    RUNG_SHAPE,
    SCRIPT_UNKNOWN,
    dominant_scripts,
    fold_key,
    scripts_present,
)

# =====================================================================================
# The URL membership rule — THE one callable
# =====================================================================================

#: The scheme / host prefixes a licence comparison ignores. One page is one page whether the
#: agent writes ``http://`` or ``https://``, with or without ``www.`` — a licence that reds
#: those variants calls a documented address fabricated, the false red this module exists to
#: remove. Only these two schemes: ``mailto:`` / ``tel:`` addresses are different resources.
_LICENCE_SCHEME_RE = re.compile(r'^https?://', re.IGNORECASE)
_LICENCE_WWW_RE = re.compile(r'^www\.', re.IGNORECASE)


def licence_form(url: str) -> str:
    """A URL's comparable IDENTITY: ``host/path[#/route][?query]``, with the cosmetic parts gone.

    Cosmetic (dropped): the scheme, a leading ``www.``, a trailing slash on the PATH *or the
    ROUTE*, a plain ``#anchor``, and the ``!`` of a hashbang route (``#!/x`` IS ``#/x``) — one
    page is one page however the agent writes it, and a licence that reds those variants calls a
    documented address fabricated.

    Kept, because each names a different resource: the QUERY (``?id=9999`` is not ``?id=1042`` —
    :func:`url_licensed` holds the asymmetric rule that governs it) and a ROUTE fragment, one
    beginning ``#/``. Hash routing is how a single-page app addresses its pages, so dropping it
    would collapse every ``…/#/asset/<id>`` to its host: two owners for one address, no misattribution
    verdict reachable, and any invented route under a documented host licensed.
    """
    base, _, fragment = url.strip().partition('#')
    path, _, query = base.partition('?')
    # The trailing slash comes off the PATH ONLY: ``/asset/?id=1`` and ``/asset?id=1`` are one
    # address, which a strip over the whole string could not see.
    path = _LICENCE_WWW_RE.sub('', _LICENCE_SCHEME_RE.sub('', path)).rstrip('/')
    # The ROUTE is a path too, so it loses its trailing slash for the SAME reason the path does:
    # ``…/#/asset/1042`` and ``…/#/asset/1042/`` are one documented page, and reading them as two
    # reds a correct citation of a route the sources carry. The hashbang spelling (``#!/…``, how
    # an older SPA writes the same route) normalizes onto the plain one rather than being read as
    # a plain anchor — read as an anchor it would be DROPPED, collapsing every hashbang route to its
    # bare host: two owners for one address, and any invented route under a documented host
    # licensed.
    routed = fragment[1:] if fragment.startswith('!') else fragment
    route = f'#{routed.rstrip("/")}' if routed.startswith('/') else ''
    return f'{path}{route}' + (f'?{query}' if query else '')


def _licence_parts(url: str) -> tuple[str, str]:
    """:func:`licence_form` split into ``(identity without the query, query)``."""
    form = licence_form(url)
    path, _, query = form.partition('?')
    return path, query


def url_licensed(url: Any, members: Iterable[str]) -> bool:
    """Whether ``members`` (literal URLs and ``<…>``/``{…}`` templates) license ``url``.

    THE membership rule, shared by the sandbox helper ``corpus_url_licensed``, the title index's
    ``url_owners``, ``urls.LicensedUrl.licenses`` and the URL rung of the ladder below. Both sides
    pass through :func:`licence_form`; ``url_matches`` then does the template-aware comparison (it
    degrades to case-insensitive equality on a placeholder-free entry, so no separate equality
    disjunct decides anything). Two spellings of this rule would let the code_eval license what
    the judge reds.

    The QUERY is ASYMMETRIC, and both directions are load-bearing:

    * the licensed form carries NO query → any query on the cited URL is licensed. A documented
      page licenses its own parameterised reads: ``?lang=en``, ``?utm_source=…``, ``?page=2`` are
      how the same page is fetched, and calling them fabrications reds a correct reply. It is
      also the only workable rule — a documentation set cannot write ``?id=<asset>`` instead,
      because the backend's URL extractor truncates a query-position placeholder to a dead
      prefix, so the "document it as a template" remedy does not exist.
    * the licensed form HAS one → the cited query must match it (literally, or through a template
      the extractor can carry). ``?id=9999`` is a different asset from the documented ``?id=1042``,
      and licensing it hands a fabricated id a clean bill.
    """
    if not isinstance(url, str):
        return False
    candidate, candidate_query = _licence_parts(url)
    if not candidate:
        return False
    for entry in members:
        member, member_query = _licence_parts(str(entry))
        if not url_matches(candidate, member):
            continue
        if not member_query or url_matches(candidate_query, member_query):
            return True
    return False


#: The DECLARED RECORD line, in the one spelling both haystack builders read it through.
#: ``- name: Ann Lee`` / ``1. name: Ann Lee`` / ``  name: Ann Lee`` are all one shape.
#:
#: The key group is ``[^\W\d][\w. -]*`` — a letter or underscore, then
#: letters/digits/underscore/dot/space/hyphen — which is the UNICODE spelling of the ASCII
#: ``[A-Za-z_][A-Za-z0-9_. -]*``, not a widening of its shape. A customer whose column headings
#: are Hebrew, Greek, Japanese or Cyrillic therefore declares records, so the identity test has
#: something to run on and the closed-class licence, with the detector ladder built on it, applies
#: to that customer. The title index still requires an English title key, so a record whose keys
#: are all in another script contributes no titled entity. Which keys CLOSE is decided separately,
#: and is not widened here.
#:
#: It lives HERE, in the vendorable module, because BOTH sides have to exclude the same lines from
#: their specificity haystack, and one regex has to decide which lines those are. A numbered-list
#: catalog (``1. name: …`` — a shape the record scanner explicitly supports) is a record and a
#: quoted key (``"name": …``) is not, on both sides, so the same document yields the same haystack
#: everywhere and a class cannot be refused on one side and activated on the other.
KEY_VALUE_LINE_RE = re.compile(r'^[ \t]*(?P<item>(?:[-*+]|\d+\.)[ \t]+)?(?P<key>[^\W\d][\w. -]*):(?:[ \t]+(?P<value>.*))?$')


def line_key(line: str) -> str:
    """The key a declared record line carries, casefolded, or ``''`` when the line is not one.

    THE reader, so the lines a class excludes from its haystack are decided by the same regex that
    parsed the record in the first place.
    """
    match = KEY_VALUE_LINE_RE.match(line)
    return match.group('key').strip().casefold() if match else ''


# =====================================================================================
# The typed record
# =====================================================================================

#: What separates a class id's family label from its key: ``field:leaders.name`` is the ``name``
#: column of the ``leaders`` family, and ``field:name`` is the UNION alias the judge is shown.
#: Read by the reader that has to SPLIT an id back apart on a record that carries no ``alias``;
#: the builders join with an f-string, because a subject is ``<entity>.<field>`` by the scan-fact
#: vocabulary's own definition, not by this module's choice.
CLOSED_CLASS_ALIAS_SEPARATOR = '.'

#: Every rung name, in the order the ladder descends — RE-EXPORTED from the module that owns the
#: vocabulary (``facts`` validates a customer-declared ``candidate.rung`` and cannot import this
#: one). Read here as the membership test that keeps an unrecognised recorded rung off ``active``.
RUNGS: tuple[str, ...] = DETECTOR_RUNGS

STATUS_ACTIVE = 'active'
STATUS_REFUSED = 'refused'
STATUS_NOT_MEASURED = 'not_measured'
#: ...and the fourth, which is a statement about the ARTIFACT rather than about the members: this
#: class was recomputed at load time because the registry carries no recorded decision (it was
#: written before detectors existed). It must never render as ``active``: the code that would have
#: measured the gates is whichever version happens to be installed at re-score time, which is
#: exactly the drift recording exists to remove.
STATUS_NOT_RECORDED = 'not_recorded'

EQUIVALENCE_URL = 'url_licensed'
EQUIVALENCE_FOLD_KEY = 'fold_key_exact'
EQUIVALENCE_NEAR_MISS = 'fold_key_near_miss'

DECIDED_BY_BACKEND = 'backend'
DECIDED_BY_CLIENT = 'client'
DECIDED_BY_OVERRIDE = 'override'

GATE_UNANIMITY = 'unanimity'
GATE_SAMPLE_FLOOR = 'sample_floor'
GATE_SPECIFICITY = 'specificity'
GATE_SCRIPT = 'script'
GATE_SILENCE = 'silence'
GATE_POSITIVE_CONTROL = 'positive_control'

#: Every gate, in the order they are measured and reported. Closed and ordered so a refusal reads
#: the same way twice and a gate added without a measurement cannot silently vanish from a record.
GATES: tuple[str, ...] = (GATE_UNANIMITY, GATE_SAMPLE_FLOOR, GATE_SPECIFICITY, GATE_SCRIPT, GATE_SILENCE, GATE_POSITIVE_CONTROL)

#: How many members a PRE-VALIDATED shape needs. Three: the shape is not being invented here, it
#: is being SELECTED from a closed registry, so the sample only has to show the class agrees with
#: one of them rather than carry enough evidence to induce a pattern.
SHAPE_MIN_MEMBERS = 3
#: How many members an INDUCED pattern needs, and from how many record families. Inducing a regex
#: from three strings is a coincidence, not a pattern — ``_CLASS_MIN_RECORDS = 3`` is the floor for
#: FORMING a class, which is a different question. Two families because a skeleton shared by one
#: document's rows is that document's formatting.
INDUCED_MIN_MEMBERS = 8
INDUCED_MIN_FAMILIES = 2
#: How many members a near-miss rung needs, and how many KEY characters each must carry. Shorter
#: members produce ``Acme`` / ``Acmes`` / ``acne`` collisions, which is a false red on ordinary prose.
NEAR_MISS_MIN_MEMBERS = 3
NEAR_MISS_MIN_KEY_CHARS = 6

#: How many lines of ordinary prose the specificity gate needs before its silence means anything.
#: One unrelated sentence is not meaningfully more evidence than none, so the floor is THREE, the
#: same as every other sample in this module and for the same reason: it refuses the degenerate
#: case without deleting the detector on a small doc set, which would be a miss bought with
#: nothing. The count rides the record either way, so a reader
#: weighing a pass against a thin haystack can see how thin it was.
SPECIFICITY_MIN_UNITS = 3

#: How many offending strings a refusal quotes. Bounded because the evidence can carry real
#: customer records, and because ten quotes of the same mistake are not ten findings.
MAX_GATE_EVIDENCE = 3
#: How much of a source line one quote carries.
GATE_EXCERPT_CHARS = 160


@dataclass(frozen=True)
class GateResult:
    """What ONE gate established, and the evidence that decided it.

    ``passed`` is THREE-valued on purpose. ``None`` is ``not_measured`` — the measurement could
    not be made (an empty specificity haystack, no declared negatives to control against) — and it
    is not ``True``. A roster document whose only lines are the records would otherwise pass
    specificity vacuously, and a registry carrying only the scan's URL channel has no units at all.
    """

    name: str
    passed: bool | None
    #: Free-form measurement, rendered beside the verdict: counts, floors, quoted offenders.
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {'passed': self.passed, **dict(self.evidence)}

    @property
    def detail(self) -> str:
        return ', '.join(f'{k}={v!r}' for k, v in self.evidence.items() if v not in (None, [], (), ''))

    def refusal(self) -> str:
        """One line naming the GATE and its measurement, for a ``NOT CHECKED IN CODE`` clause —
        where the verdict is already in the surrounding sentence and repeating it reads as noise."""
        return f'{self.name}' + (f' ({self.detail})' if self.detail else '')


@dataclass(frozen=True)
class Detector:
    """The ladder's typed decision about ONE closed class.

    Recorded as data rather than inferred by any consumer: the judge prompt states
    ``checked in code by <rule>`` only when :attr:`active`, the scan's ``closed_class_detectors``
    check joins on it, and the sandbox abstains on anything else.
    """

    rung: str = RUNG_NONE
    status: str = STATUS_NOT_MEASURED
    #: Human-readable name of the equivalence that fired — what a report prints after
    #: "checked in code by". Empty when nothing fired.
    rule: str = ''
    #: The induced or overridden regex, or ``None``. Never a shape-registry pattern: those are
    #: named by :attr:`rule`, and publishing their source would invite hand-editing a closed
    #: vocabulary through a data file.
    pattern: str | None = None
    equivalence: str = ''
    decided_by: str = ''
    gates: tuple[GateResult, ...] = ()
    #: Why the rung that was tried did not activate — the FIRST failing gate, in words. Empty
    #: when active. Distinct from the gate list because a reader acts on one sentence.
    reason: str = ''
    #: What the ladder tried before this, one line per rung, so a refusal names the whole descent
    #: rather than only its last step.
    descent: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.status == STATUS_ACTIVE

    def clause(self) -> str:
        """The one clause a licence line gains: what checked this class, or that nothing did."""
        if self.active:
            return f'checked in code by {self.rule}'
        return f'NOT CHECKED IN CODE ({self.status}: {self.reason or "no rung was measured"})'

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'rung': self.rung, 'status': self.status}
        if self.rule:
            out['rule'] = self.rule
        out['pattern'] = self.pattern
        if self.equivalence:
            out['equivalence'] = self.equivalence
        if self.decided_by:
            out['decided_by'] = self.decided_by
        if self.reason:
            out['reason'] = self.reason
        if self.gates:
            out['gates'] = {g.name: g.as_dict() for g in self.gates}
        if self.descent:
            out['descent'] = list(self.descent)
        return out


def read_detector(raw: Any) -> Detector:
    """Coerce a recorded ``detector:`` mapping back into a :class:`Detector`. Total on any input.

    A malformed or absent record yields :data:`STATUS_NOT_RECORDED`, never a default that reads as
    ``active``: this runs over a ``source_registry.yaml`` written by an older backend, and the one
    thing an unreadable record must not do is claim a check ran.
    """
    if not isinstance(raw, Mapping):
        return Detector(status=STATUS_NOT_RECORDED, reason='the registry carries no recorded detector decision for this class')
    status = str(raw.get('status') or '')
    if status not in (STATUS_ACTIVE, STATUS_REFUSED, STATUS_NOT_MEASURED, STATUS_NOT_RECORDED):
        return Detector(status=STATUS_NOT_RECORDED, reason=f'the recorded detector status {status!r} is not one this build knows')
    gates_raw = raw['gates'] if isinstance(raw.get('gates'), Mapping) else {}
    gates = tuple(
        # ``passed`` is coerced, not carried. It arrives from YAML, where ``passed: yes`` is a
        # string and ``'yes'`` is neither True nor None — and this is the one function whose
        # contract is that an unreadable record never claims a check ran. Anything that is not
        # a real bool reads as NOT MEASURED, which is the refusing direction.
        GateResult(
            name=name,
            passed=body['passed'] if isinstance(body.get('passed'), bool) else None,
            evidence={k: v for k, v in body.items() if k != 'passed'},
        )
        for name in GATES
        if isinstance(body := gates_raw.get(name), Mapping)
    )
    rung = str(raw.get('rung') or RUNG_NONE)
    detector = Detector(
        rung=rung if rung in (*RUNGS, RUNG_NONE) else RUNG_NONE,
        status=status,
        rule=str(raw.get('rule') or ''),
        pattern=str(raw['pattern']) if raw.get('pattern') else None,
        equivalence=str(raw.get('equivalence') or ''),
        decided_by=str(raw.get('decided_by') or ''),
        gates=gates,
        reason=str(raw.get('reason') or ''),
        descent=tuple(str(d) for d in raw.get('descent') or () if str(d).strip()),
    )
    return detector if detector.status != STATUS_ACTIVE else _honoured_active(detector)


def _honoured_active(detector: Detector) -> Detector:
    """An ``active`` record, or the same record DEMOTED when this build cannot honour the claim.

    ``active`` has to mean a check actually runs, and three recorded shapes break that promise
    while still reading as a clean licence — each of them reachable from a registry written by a
    different build, or by a serializer that dropped a field:

    * a ``shape`` rung naming a rule this build's registry does not carry (the vocabulary is
      closed within one build, and builds differ);
    * an ``induced`` rung whose ``pattern`` is missing or does not compile;
    * any rung the reader could not recognise, which lands on ``none``.

    Demoted to :data:`STATUS_NOT_RECORDED` rather than ``refused``: nothing about the CLASS was
    refused — this build simply cannot re-run what was recorded, which is a fact about the
    artifact. The alternative is what this returns you to: a judge told "checked in code by
    &lt;rule&gt;" beside a detection block reporting ``0 outside … of 0 value(s) lifted``, which is
    also exactly what an honest clean reply looks like.
    """
    if detector.rung == RUNG_SHAPE and detector.rule not in {shape.name for shape in SHAPES}:
        reason = f'the recorded rule {detector.rule!r} is not one this build carries, so the decision cannot be re-run'
    elif detector.rung == RUNG_INDUCED and not _compilable(detector.pattern):
        reason = f'the recorded induced pattern {detector.pattern!r} is absent or does not compile, so the decision cannot be re-run'
    elif detector.rung not in RUNGS:
        reason = f'the record says active but names no rung this build can run (rung={detector.rung!r})'
    elif any(gate.passed is not True for gate in detector.gates):
        failed = next(gate for gate in detector.gates if gate.passed is not True)
        reason = f'the record says active while its own {failed.name} gate does not say passed'
    else:
        return detector
    return replace(detector, status=STATUS_NOT_RECORDED, rule='', reason=reason)


def _compilable(pattern: str | None) -> bool:
    if not pattern:
        return False
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


# =====================================================================================
# Rung 1 — the closed shape registry
# =====================================================================================


@dataclass(frozen=True)
class Shape:
    """One pre-validated form: how to recognise a member, how to lift candidates out of prose,
    and how to decide membership.

    Selection is by PROPERTY — every member must satisfy :attr:`matches` — never by class name.
    The registry is a closed vocabulary; a customer-supplied pattern is
    rung 2, where the sample floors and the gates apply.
    """

    name: str
    #: Whether ONE member has this form.
    matches: Callable[[str], bool]
    #: Every candidate value of this form in a text, in order. The detector's whole reading side.
    lift: Callable[[str], list[str]]
    #: Whether ``members`` license ``value``. ``Any`` on the value because a reply carries whatever
    #: the extractor found and ``url_licensed`` defends against a non-string; the ARITY is what
    #: matters here — a one-argument callable raises ``TypeError`` from inside a gate,
    #: mid-ladder, naming neither the class nor the rung.
    licenses: Callable[[Any, Iterable[str]], bool]
    equivalence: str
    #: Whether this rule was INFERRED FROM THESE MEMBERS (an induced skeleton, a customer's
    #: declared pattern) rather than selected from the closed shape vocabulary. The script gate
    #: asks this, not the rung name: a pre-validated form is a protocol that carries its own
    #: alphabet, while a rule read off the members can be an ``[A-Z]{2}`` induced over a Hebrew
    #: corpus that then reds ``OK`` and ``HR`` in an English reply.
    derived_from_members: bool = False


def _url_member(value: str) -> bool:
    """A URL, INCLUDING a documented template.

    The template half is not a nicety. The live ``url`` class is scan-sourced and the scan channel
    carries template members (``https://docs.example.com/<page>``), so a predicate that rejects
    ``<…>`` fails unanimity on every scan-sourced class — which would delete the URL detector.
    """
    return extract_urls(value) == [value] or is_url_template(value)


_EMAIL_RE = re.compile(r'^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]{2,}$')
_EMAIL_LIFT_RE = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
#: E.164 and the ordinary international spellings around it. Deliberately NOT a national-format
#: matcher: "(555) 1234" is indistinguishable from a citation and a room number, and a phone shape
#: that fires on those reds ordinary prose.
_PHONE_RE = re.compile(r'^\+\d[\d\s().-]{6,19}\d$')
_PHONE_LIFT_RE = re.compile(r'\+\d[\d\s().-]{6,19}\d')
_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
_UUID_LIFT_RE = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE)
#: A path with at least one separator and an extension or a leading root/dot — enough structure
#: that a bare English word can never be one.
_PATH_RE = re.compile(r'^(?:\.{0,2}/)?(?:[\w.+-]+/)+[\w.+-]+$')
_PATH_LIFT_RE = re.compile(r'(?<![\w/])(?:\.{0,2}/)?(?:[\w.+-]+/)+[\w.+-]+(?![\w/])')
_HANDLE_RE = re.compile(r'^@[\w.-]{2,}$')
_HANDLE_LIFT_RE = re.compile(r'(?<![\w@])@[\w.-]{2,}')
_VERSION_RE = re.compile(r'^v?\d+\.\d+(?:\.\d+)*$', re.IGNORECASE)
_VERSION_LIFT_RE = re.compile(r'(?<![\w.])v?\d+\.\d+(?:\.\d+)*(?![\w.])', re.IGNORECASE)
_CURRENCY_RE = re.compile(r'^[$€£₪¥]\s?\d[\d,]*(?:\.\d{1,2})?$')
_CURRENCY_LIFT_RE = re.compile(r'[$€£₪¥]\s?\d[\d,]*(?:\.\d{1,2})?')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_DATE_LIFT_RE = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
#: A NUMERIC id is at least four digits. Below that every page number, count and year in the
#: sources is a candidate, and the specificity gate would refuse the rung anyway — refusing it
#: here means the refusal names the shape rather than the corpus.
_NUMERIC_ID_RE = re.compile(r'^\d{4,}$')
_NUMERIC_ID_LIFT_RE = re.compile(r'(?<!\d)\d{4,}(?!\d)')


def _fold_key_licenses(value: Any, members: Iterable[str]) -> bool:
    """Exact membership after :func:`fold_key` — the ladder's default equivalence.

    ONE spelling, shared by the shape registry, the induced rung and the near-miss rung: a
    normalisation has to agree with itself, or a class licenses under one rung what it reds under
    another.
    """
    key = fold_key(str(value))
    return bool(key) and any(fold_key(str(m)) == key for m in members)


def _regex_shape(name: str, member_re: re.Pattern[str], lift_re: re.Pattern[str]) -> Shape:
    """A shape whose membership is EXACT after :func:`fold_key` — the ladder's default equivalence.

    Every form here folds case, including the two where a specification says otherwise: an email
    address's local part is case-sensitive per RFC and a file path is case-sensitive on most
    filesystems. Neither customer document means it, and a licence that reds ``Support@`` against a
    documented ``support@`` calls a documented address fabricated — the false red this whole module
    exists to remove.
    """
    return Shape(
        name=f'the {name} shape',
        matches=lambda value: bool(member_re.match(value.strip())),
        lift=lambda text: lift_re.findall(text or ''),
        licenses=_fold_key_licenses,
        equivalence=EQUIVALENCE_FOLD_KEY,
    )


#: The closed shape vocabulary, tried in order. ``url`` leads because it is the only entry whose
#: equivalence is not ``fold_key``; the rest are ordered most-structured first, so a value that
#: satisfies two predicates is attributed to the tighter one.
SHAPES: tuple[Shape, ...] = (
    # NAMED FOR THE CALLABLE, because that is what the licence line claims: "checked in code by
    # url_licensed" is checkable against this file, where "checked in code by url" is a category.
    Shape(name='url_licensed', matches=_url_member, lift=lambda text: extract_urls(text or ''), licenses=url_licensed, equivalence=EQUIVALENCE_URL),
    _regex_shape('email', _EMAIL_RE, _EMAIL_LIFT_RE),
    _regex_shape('uuid', _UUID_RE, _UUID_LIFT_RE),
    _regex_shape('phone', _PHONE_RE, _PHONE_LIFT_RE),
    _regex_shape('date', _DATE_RE, _DATE_LIFT_RE),
    _regex_shape('currency_amount', _CURRENCY_RE, _CURRENCY_LIFT_RE),
    _regex_shape('version', _VERSION_RE, _VERSION_LIFT_RE),
    _regex_shape('handle', _HANDLE_RE, _HANDLE_LIFT_RE),
    _regex_shape('file_path', _PATH_RE, _PATH_LIFT_RE),
    _regex_shape('numeric_id', _NUMERIC_ID_RE, _NUMERIC_ID_LIFT_RE),
)


def matching_shape(members: Sequence[str]) -> Shape | None:
    """The first registry shape EVERY member has, or ``None``.

    Unanimity is applied here rather than reported as a gate result for the shape rung, because a
    shape only half the members have is not a candidate at all — there is nothing to measure the
    other five gates against. The gate is still recorded (see :func:`derive_detector`) so the
    record says which measurement refused.
    """
    return next((shape for shape in SHAPES if members and all(shape.matches(m.strip()) for m in members if m.strip())), None)


# =====================================================================================
# Rung 2 — the induced pattern
# =====================================================================================

#: The character classes induction works over. Deliberately coarse: induction is over character
#: CLASSES and lengths, never over a dictionary, so ``AB-1234`` and ``CD-5678`` induce
#: ``[A-Z]{2}-[0-9]{4}`` and ``Rhea Castellan`` / ``Alec Brandt`` induce nothing usable — which is
#: the right answer, because names belong to the near-miss rung.
_CLASS_UPPER = 'A-Z'
_CLASS_LOWER = 'a-z'
_CLASS_DIGIT = '0-9'


def _char_class(char: str) -> str | None:
    if char.isdigit() and char.isascii():
        return _CLASS_DIGIT
    if char.isascii() and char.isupper():
        return _CLASS_UPPER
    if char.isascii() and char.islower():
        return _CLASS_LOWER
    return None


def _skeleton(value: str) -> tuple[tuple[str, int], ...] | None:
    """``value`` as ``((class-or-literal, run length), …)``, or ``None`` when unrepresentable.

    A run of one character class collapses to one entry; anything else (a separator, punctuation,
    a non-ASCII letter) is kept LITERALLY, so ``KS-12/A`` induces its own separators rather than a
    wildcard. ``None`` for a value carrying a character no class covers and that is not a plain
    ASCII separator — a Hebrew or Han member cannot be induced over these classes, and returning a
    skeleton that ignored it would induce a pattern that matches the wrong thing.
    """
    out: list[tuple[str, int]] = []
    for char in value:
        cls = _char_class(char)
        if cls is None:
            if not char.isascii() or char.isspace():
                # A non-ASCII letter has no class here; a SPACE makes a token boundary, which
                # ``_token_skeleton`` handles instead. Either way this value is not inducible.
                return None
            cls = f'\\{char}' if char in '.^$*+?()[]{}|\\-/' else char
        if out and out[-1][0] == cls:
            out[-1] = (cls, out[-1][1] + 1)
        else:
            out.append((cls, 1))
    return tuple(out)


def _skeleton_pattern(skeleton: Sequence[tuple[str, int]]) -> str:
    parts = []
    for cls, run in skeleton:
        atom = f'[{cls}]' if cls in (_CLASS_UPPER, _CLASS_LOWER, _CLASS_DIGIT) else cls
        parts.append(f'{atom}{{{run}}}' if run > 1 else atom)
    return '^' + ''.join(parts) + '$'


def induce_pattern(members: Sequence[str]) -> str | None:
    """A regex EVERY member matches, induced from the members themselves, or ``None``.

    Two shapes are inducible and both are structural:

    * one shared skeleton — every member is the same run of character classes and literals
      (``AB-1234``);
    * one shared TOKEN skeleton — every member is the same number of whitespace-separated tokens,
      each with its own shared skeleton. This is what makes a fixed-arity code with a space in it
      work without loosening the single-skeleton rule.

    ``None`` when the members disagree, which is the common and correct outcome: a roster of human
    names has neither shape, and inducing ``[A-Z][a-z]{2,8} [A-Z][a-z]{2,8}`` from four of them
    would fire on every capitalised pair in the customer's prose.
    """
    cleaned = [m.strip() for m in members if m.strip()]
    if not cleaned:
        return None
    skeletons = [_skeleton(m) for m in cleaned]
    if all(s is not None for s in skeletons) and len(set(skeletons)) == 1:
        return _skeleton_pattern(skeletons[0])  # type: ignore[arg-type]
    token_rows = [m.split() for m in cleaned]
    arity = {len(row) for row in token_rows}
    if len(arity) != 1 or arity == {1}:
        return None
    columns = list(zip(*token_rows, strict=True))
    per_column: list[tuple[tuple[str, int], ...]] = []
    for column in columns:
        column_skeletons = {_skeleton(token) for token in column}
        if len(column_skeletons) != 1 or None in column_skeletons:
            return None
        per_column.append(column_skeletons.pop())  # type: ignore[arg-type]
    return '^' + r'\s+'.join(_skeleton_pattern(s)[1:-1] for s in per_column) + '$'


def _induced_shape(pattern: str) -> Shape:
    """A rung-2 shape over one induced or customer-supplied regex.

    The lift side anchors the compiled pattern at token boundaries rather than searching for it
    unanchored: an induced ``^[A-Z]{2}-[0-9]{4}$`` must not lift ``XAB-12345`` out of a longer
    token and report it as an outside-class value.
    """
    member_re = re.compile(pattern)
    body = pattern[1:-1] if pattern.startswith('^') and pattern.endswith('$') else pattern
    lift_re = re.compile(rf'(?<![^\W_])(?:{body})(?![^\W_])')

    return Shape(
        name=f'pattern {pattern}',
        matches=lambda value: bool(member_re.match(value.strip())),
        lift=lambda text: [m.group(0) for m in lift_re.finditer(text or '')],
        licenses=_fold_key_licenses,
        equivalence=EQUIVALENCE_FOLD_KEY,
        # INFERRED from these values (or declared for them), so the script gate has to ask whether
        # its alphabet agrees with the sources' and the replies'.
        derived_from_members=True,
    )


# =====================================================================================
# Rung 3 — near miss
# =====================================================================================


def _damerau_levenshtein(a: str, b: str, budget: int) -> int:
    """Optimal string alignment distance, capped at ``budget + 1``.

    Optimal-string-alignment rather than unrestricted Damerau-Levenshtein: the two differ only on
    a string with two transpositions sharing a character, which cannot arise inside the budgets
    here (``max(1, len // 8)`` over a member of at least six key characters), and OSA is the
    version with a bounded early exit. The cap is what keeps this affordable over every member of
    every class for every candidate token in a reply.
    """
    if abs(len(a) - len(b)) > budget:
        return budget + 1
    previous_previous: list[int] = []
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i, *([0] * len(b))]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous_previous[j - 2] + cost)
        if min(current) > budget:
            return budget + 1
        previous_previous, previous = previous, current
    return previous[-1]


def near_miss_budget(member_key: str) -> int:
    """How many edits from a member still counts as that member CORRUPTED, not a different value."""
    return max(1, len(member_key) // 8)


def _trim_to_key_chars(window: str) -> str:
    """``window`` with its leading and trailing non-key characters removed.

    A window is cut on whitespace, so it routinely carries the sentence's punctuation:
    ``"Rhea Castellan."`` folds to the same key as the member and is correctly licensed, but the
    SURFACE is what gets quoted to a judge told to "cite the value", and quoting a value the reply
    does not contain is how a correct citation reads as a near miss of itself.
    """
    start, end = 0, len(window)
    while start < end and fold_key(window[start]) == '':
        start += 1
    while end > start and fold_key(window[end - 1]) == '':
        end -= 1
    return window[start:end]


def _charset_mask(key: str) -> int:
    """A 64-bit set of the characters in ``key``, for the cheap pruning filter above."""
    mask = 0
    for char in key:
        mask |= 1 << (ord(char) & 63)
    return mask


def _near_a_member(key: str, by_length: Mapping[int, Sequence[tuple[str, int, int]]]) -> bool:
    """Whether ``key`` is within its budget of any member key, using the length + mask index."""
    key_mask = _charset_mask(key)
    for member_key, budget, member_mask in by_length.get(len(key), ()):
        if ((key_mask ^ member_mask).bit_count()) > 2 * budget:
            continue
        if _damerau_levenshtein(key, member_key, budget) <= budget:
            return True
    return False


def _near_miss_windows(text: str, sizes: Iterable[int]) -> Iterable[str]:
    """Whitespace-token windows of ``text`` at each token count in ``sizes``.

    Windows rather than a scan over every substring: a member is a phrase ("Alec Brandt"), and the
    corrupted form the rung exists to catch has the same token count or one adjacent to it. Every
    window is compared through :func:`fold_key`, so punctuation between the tokens never decides.
    """
    tokens = (text or '').split()
    for size in sorted({s for s in sizes if s > 0}):
        for start in range(max(0, len(tokens) - size + 1)):
            yield ' '.join(tokens[start : start + size])


def _near_miss_shape(members: Sequence[str]) -> Shape:
    """The rung-3 shape: membership is exact after :func:`fold_key`; DETECTION is the distance rule.

    ``lift`` returns the windows that ARE a member or are one edit from one — nothing else. That is
    the whole point of the rung: it never asks "which token is an identifier", the question that
    cannot be answered without inventing the thing being policed, so it can only ever report a
    corruption of something the sources actually enumerate. The exact members come back too, for
    the exoneration side; see ``_lift`` for why that can never manufacture an accusation.
    """
    keys = {fold_key(m) for m in members if fold_key(m)}
    sizes = {len(m.split()) for m in members if m.strip()}
    sizes |= {s + 1 for s in sizes if s} | {s - 1 for s in sizes if s > 1}
    # THE INDEX. This lift runs per judge pass and per code_eval policy, over every reply, and the
    # naive form is a full Damerau-Levenshtein against every member for every window, which is slow
    # enough that a judge times out and lands as an inconclusive verdict. Two sound filters avoid it:
    #
    # * bucket member keys by LENGTH, so only keys within the budget's length band are considered;
    # * a 64-bit character-set mask per key. Each edit changes the character set by at most one in
    #   each direction, so a symmetric difference wider than 2*budget cannot be reached in budget
    #   edits. Necessary, never sufficient — a survivor still runs the real distance.
    #
    # Both are pure pruning: the verdict is identical, asserted over randomised corpora.
    budgets = {key: near_miss_budget(key) for key in keys}
    by_length: dict[int, list[tuple[str, int, int]]] = {}
    for key, budget in budgets.items():
        entry = (key, budget, _charset_mask(key))
        for length in range(len(key) - budget, len(key) + budget + 1):
            by_length.setdefault(length, []).append(entry)

    def _lift(text: str) -> list[str]:
        """Windows that are a member, or CLOSE to one. Nothing else, ever.

        The exact-member windows are lifted so the two-sided report has an exoneration side: a
        reply that cited four people correctly reads differently from one that named nobody, and the
        report renders them differently. They can never become an outside-class value — ``licenses``
        is exact-after-``fold_key`` over the same keys — so lifting them costs nothing and adds the
        only evidence the rung can honestly give that it read anything.
        """
        found: dict[str, None] = {}
        # Memoised per DISTINCT key: overlapping windows of one reply repeat heavily (12k windows
        # collapse to 7.5k keys on a 4000-word reply), and the answer depends only on the key.
        verdict: dict[str, bool] = {}
        for raw in _near_miss_windows(text, sizes):
            window = _trim_to_key_chars(raw)
            key = fold_key(window)
            if not key or len(key) < NEAR_MISS_MIN_KEY_CHARS:
                continue
            hit = verdict.get(key)
            if hit is None:
                hit = key in keys or _near_a_member(key, by_length)
                verdict[key] = hit
            if hit:
                found.setdefault(window, None)
        return list(found)

    return Shape(
        name='near-miss on the enumerated members',
        matches=lambda value: bool(value.strip()),
        lift=_lift,
        licenses=_fold_key_licenses,
        equivalence=EQUIVALENCE_NEAR_MISS,
    )


def _member_scripts(members: Sequence[str]) -> set[str]:
    scripts: set[str] = set()
    for member in members:
        scripts |= scripts_present(member)
    return scripts


# =====================================================================================
# The six gates
# =====================================================================================


@dataclass(frozen=True)
class LadderInput:
    """Everything the gates measure against, gathered by the caller that owns the corpus.

    Split from the members so the two SIDES can differ honestly: the client measures over every
    ``CorpusUnit.text`` under the scanned roots, the backend over every registered unit's text.
    They are not the same bytes, and a detector active on one and refused on the other is a
    pre-generation finding rather than a contradiction to paper over.
    """

    #: Ordinary prose the detector must be SILENT on outside the member lines: the specificity and
    #: silence haystacks. Empty is ``not_measured``, never ``passed``.
    haystack: Sequence[str] = ()
    #: The lines the class was LIFTED from, excluded from the haystack — a roster document whose
    #: only lines are the records would otherwise pass specificity vacuously.
    member_lines: Sequence[str] = ()
    #: Literals the author asserts are NOT members, from the scan fact's ``detector.negatives``.
    #: The positive control: specificity proves the detector fires on nothing wrong, this proves
    #: it fires on something right.
    negatives: Sequence[str] = ()
    #: How many record families contributed members — the induced rung's second floor.
    families: int = 1
    #: Scripts the agent's declared reply languages are written in, when the caller knows them.
    reply_scripts: Sequence[str] = ()
    decided_by: str = DECIDED_BY_BACKEND


def _excerpt(line: str) -> str:
    text = ' '.join((line or '').split())
    return text[:GATE_EXCERPT_CHARS] + ('…' if len(text) > GATE_EXCERPT_CHARS else '')


def _prose_haystack(source: LadderInput) -> list[str]:
    """The haystack with the member lines removed, exactly once per line."""
    excluded = {' '.join(line.split()) for line in source.member_lines if line.strip()}
    return [line for line in source.haystack if line.strip() and ' '.join(line.split()) not in excluded]


#: Why a rung had no candidate rule at all, per rung. Stated rather than generated so a refusal a
#: customer reads names what to DO about it, not merely that nothing fit.
_NO_RULE_REASON: dict[str, str] = {
    RUNG_SHAPE: 'no pre-validated shape fits all {n} member(s)',
    RUNG_INDUCED: 'the {n} member(s) share no skeleton to induce a pattern from',
    RUNG_NEAR_MISS: (
        'a distance rule cannot decide these {n} member(s): a pure-letter member is a WORD, and one edit from a word is its plural, '
        'its genitive or its possessive — ordinary correct prose. Only identifier-shaped members (carrying a digit) have a one-edit '
        'neighbourhood empty of correct language'
    ),
}


def _gate_unanimity(shape: Shape | None, members: Sequence[str], rung: str) -> GateResult:
    if shape is None:
        return GateResult(GATE_UNANIMITY, False, {'reason': _NO_RULE_REASON[rung].format(n=len(members))})
    outliers = [m for m in members if m.strip() and not shape.matches(m.strip())][:MAX_GATE_EVIDENCE]
    return GateResult(GATE_UNANIMITY, not outliers, {'rule': shape.name, **({'outliers': outliers} if outliers else {})})


def _gate_sample_floor(rung: str, members: Sequence[str], families: int, *, declared: bool = False) -> GateResult:
    """Enough members to justify the rule — and the floor depends on where the rule CAME FROM.

    ``declared`` is the customer override: a pattern the customer wrote, with a source line, in their own
    repository. It is held to the PRE-VALIDATED floor rather than the induction floor, because the
    induction floor exists to justify inferring a skeleton FROM the members ("inducing a regex from
    three strings is a coincidence, not a pattern") and a declared pattern is not inferred from
    them at all. Holding it to the induction floor would refuse a customer's own declaration for a
    reason that has nothing to do with their pattern — and would disable the escape hatch exactly
    where it is needed, since a SKU catalog is usually one file, which is one record family. The
    other five gates still run against it unchanged; specificity is the one that actually decides
    whether a declared pattern is too loose.
    """
    counted = len([m for m in members if m.strip()])
    if rung == RUNG_SHAPE or declared:
        return GateResult(
            GATE_SAMPLE_FLOOR, counted >= SHAPE_MIN_MEMBERS, {'members': counted, 'floor': SHAPE_MIN_MEMBERS, **({'declared': True} if declared else {})}
        )
    if rung == RUNG_INDUCED:
        passed = counted >= INDUCED_MIN_MEMBERS and families >= INDUCED_MIN_FAMILIES
        return GateResult(
            GATE_SAMPLE_FLOOR, passed, {'members': counted, 'floor': INDUCED_MIN_MEMBERS, 'families': families, 'family_floor': INDUCED_MIN_FAMILIES}
        )
    long_enough = [m for m in members if len(fold_key(m)) >= NEAR_MISS_MIN_KEY_CHARS]
    return GateResult(
        GATE_SAMPLE_FLOOR,
        len(long_enough) >= NEAR_MISS_MIN_MEMBERS and len(long_enough) == counted,
        {'members': counted, 'long_enough': len(long_enough), 'floor': NEAR_MISS_MIN_MEMBERS, 'key_chars': NEAR_MISS_MIN_KEY_CHARS},
    )


def _non_members(shape: Shape, members: Sequence[str], prose: Sequence[str]) -> list[str]:
    """Everything the rule lifts out of the sources' ordinary prose that the class does not license.

    ONE sweep, read by BOTH address gates. They apply the same predicate to the same bytes, so a
    second sweep could not return a different verdict and would walk the whole haystack twice per
    class.

    Bounded: the verdict is decided by the first offender and the report quotes three. Collecting
    every distinct one over a real catalog is a linear scan per value for evidence nothing prints.
    """
    lifted: list[str] = []
    for line in prose:
        for value in shape.lift(line):
            if not shape.licenses(value, members) and value not in lifted:
                lifted.append(value)
                if len(lifted) >= MAX_GATE_EVIDENCE:
                    return lifted
    return lifted


def _too_thin(prose: Sequence[str], gate: str, what: str) -> GateResult | None:
    """``not_measured`` when there is too little prose for this gate's silence to mean anything.

    One unrelated sentence is not meaningfully more evidence than none, and a rule activated on it
    can fire on the customer's replies. Both address gates answer to the floor, because recording
    ``passed: True`` for silence over a one-line haystack is the vacuous pass this module's whole
    posture is against.
    """
    if len(prose) >= SPECIFICITY_MIN_UNITS:
        return None
    return GateResult(gate, None, {'haystack_units': len(prose), 'floor': SPECIFICITY_MIN_UNITS, 'reason': what})


def _gate_specificity(shape: Shape, members: Sequence[str], prose: Sequence[str]) -> GateResult:
    """Is the rule TOO LOOSE — does it fire on the sources' ORDINARY PROSE?

    The haystack EXCLUDES the lines the class was lifted from, which is what makes this a question
    about the rule rather than about the enumeration: firing on a record line is what the rule is
    for. An offender here is text the customer wrote that has nothing to do with the class, and no
    number of added members can fix it — the rule has to be tightened or refused.
    """
    if (thin := _too_thin(prose, GATE_SPECIFICITY, 'too little prose outside the member lines to establish that this rule does not over-fire')) is not None:
        return thin
    offenders = _non_members(shape, members, prose)
    return GateResult(
        GATE_SPECIFICITY,
        not offenders,
        {
            'haystack_units': len(prose),
            **(
                {
                    'fires_on': offenders,
                    'quoted': _quoting(prose, offenders),
                    'reason': 'this rule fires on the ordinary prose of the sources, so a match in a reply means nothing',
                }
                if offenders
                else {}
            ),
        },
    )


def _gate_silence(shape: Shape, members: Sequence[str], source_lines: Sequence[str]) -> GateResult:
    """Is the CLASS EXHAUSTIVE — does lifting from the WHOLE source return only members?

    A different haystack, which is what makes it a different question: this one KEEPS the record
    lines. So it sees a value the sources declare and the enumeration does not carry — an address
    the URL index dropped when it overflowed, a column the record scan did not collect — which
    specificity cannot, because it excludes exactly those lines.

    Both can fail on one corpus and they mean different things. Specificity first in :data:`GATES`
    order, because a rule that fires on ordinary prose is not worth asking exhaustiveness about.
    """
    if not source_lines:
        return GateResult(GATE_SILENCE, None, {'haystack_units': 0, 'reason': 'no source text to lift from'})
    lifted = _non_members(shape, members, source_lines)
    return GateResult(
        GATE_SILENCE,
        not lifted,
        {
            'haystack_units': len(source_lines),
            **(
                {
                    'lifted_non_members': lifted,
                    'quoted': _quoting(source_lines, lifted),
                    'reason': 'the sources declare values of this class that the enumeration does not list, so either this rule '
                    'over-lifts or the class was never exhaustive',
                }
                if lifted
                else {}
            ),
        },
    )


def _quoting(prose: Sequence[str], offenders: Sequence[str]) -> list[str]:
    """The source lines that carry the offenders, so "too loose" is actionable rather than asserted."""
    return [_excerpt(line) for line in prose if any(value in line for value in offenders)][:MAX_GATE_EVIDENCE]


#: Why the reply half of the script gate is unmeasured when the declared language is unplaceable.
_UNPLACED_REPLY_LANGUAGE = 'the declared reply language maps to no writing system this build classifies — declare it as an ISO 639 code (he, ja, ar, …)'


def _gate_script(shape: Shape, members: Sequence[str], prose: Sequence[str], reply_scripts: Sequence[str]) -> GateResult:
    """A rule DERIVED FROM THE MEMBERS must agree with the scripts the sources and replies are in.

    Only a derived rule: a pre-validated shape is a protocol (a URL, an email) that carries its own
    alphabet, and a distance rule over identifiers has no alphabet to disagree with. Asked of
    ``shape.derived_from_members`` rather than of the rung name — the property is "was this inferred
    from these values", which is true of induction and of a customer's declared pattern, and the
    rung test was right only because an override happens to be routed through ``RUNG_INDUCED``.

    An ``[A-Z]{2}`` induced over a Hebrew corpus fires on nothing in the sources, passes specificity
    vacuously, and then reds ``OK``, ``AI``, ``HR`` and ``PM`` in an English reply — which is why
    the DECLARED reply scripts are part of this and not only the corpus census.
    """
    if not shape.derived_from_members:
        return GateResult(GATE_SCRIPT, True, {'reason': f'{shape.name} carries its own alphabet rule'})
    member_scripts_found = _member_scripts(members)
    source_scripts = dominant_scripts('\n'.join(prose))
    if not source_scripts:
        return GateResult(
            GATE_SCRIPT,
            None,
            {
                'member_scripts': sorted(member_scripts_found),
                'source_scripts': [],
                'reason': 'the sources carry too little scripted prose to name a dominant script',
            },
        )
    unshared = sorted(member_scripts_found - source_scripts)
    declared_reply = {str(s) for s in reply_scripts if str(s).strip()}
    # A declared language nobody could place is NOT a script the sources lack: comparing it against
    # the census would refuse on a name, and dropping it silently would read as a full pass. The
    # reply half is UNMEASURED, and the gate says so — a real disagreement still refuses first.
    unplaced = SCRIPT_UNKNOWN in declared_reply
    declared_reply.discard(SCRIPT_UNKNOWN)
    missing_reply = sorted(declared_reply - source_scripts)
    refused = bool(unshared or missing_reply)
    return GateResult(
        GATE_SCRIPT,
        False if refused else (None if unplaced else True),
        {
            'member_scripts': sorted(member_scripts_found),
            'source_scripts': sorted(source_scripts),
            # STATED, not implied. With no declared reply scripts the second half of this gate
            # measured nothing, and a record that does not say so reads as a full pass.
            'reply_scripts': sorted(declared_reply) if declared_reply else (SCRIPT_UNKNOWN if unplaced else 'not supplied to this side'),
            **({'members_outside_sources': unshared} if unshared else {}),
            **({'reply_scripts_absent_from_sources': missing_reply} if missing_reply else {}),
            **({'reply_language_unplaced': _UNPLACED_REPLY_LANGUAGE} if unplaced else {}),
        },
    )


#: How many mutations :func:`_synthesise_decoy` tries before giving up, and how many MEMBERS it
#: tries them on. More than one member because a single one can be pathological — a template
#: covering a whole path segment licenses every within-segment mutation of itself.
_DECOY_ATTEMPTS = 8
_DECOY_BASES = 3


def _mutate_within_class(value: str, step: int) -> str | None:
    """``value`` with ONE character replaced by a different character of the SAME class.

    Same class, so the mutant still satisfies whatever predicate selected the rung — which is the
    whole point: a decoy the detector cannot even lift proves nothing about whether it reads text.
    ``step`` walks the position from the end, so successive attempts differ.
    """
    positions = [i for i, char in enumerate(value) if _char_class(char) is not None]
    if not positions:
        return None
    index = positions[-1 - (step % len(positions))]
    char = value[index]
    if char.isdigit():
        replacement = str((int(char) + 1) % 10)
    else:
        base = 'A' if char.isupper() else 'a'
        replacement = chr(ord(base) + 1) if char == base else chr(ord(char) - 1)
    return value[:index] + replacement + value[index + 1 :]


def _decoy_candidates(base: str) -> Iterable[str]:
    """Every mutation worth trying, ordered so the cheapest form-preserving one comes first.

    Three strategies, tried in order, because no single one survives every shape:

    * a character stepped WITHIN its own class — the only strategy a regex-shaped rung can accept,
      since appending anything breaks the pattern that selected it;
    * an extra PATH SEGMENT — the only strategy that escapes a URL template, because a template
      placeholder covers exactly one segment, so every within-segment mutation of
      ``docs.example.com/<page>`` is still licensed by it;
    * a DOUBLED final character — one edit, inside the near-miss budget by construction.

    Offered to every rung. The caller keeps the first candidate the shape can LIFT and does not
    LICENSE, which is the property that decides; keying the strategies on the rung name would be a
    category test that removes working candidates.
    """
    for step in range(_DECOY_ATTEMPTS):
        if (mutated := _mutate_within_class(base, step)) is not None:
            yield mutated
    yield from (base.rstrip('/') + '/plurai-decoy' * n for n in range(1, 3))
    yield from (base + base[-1] * n for n in range(1, 3))


def _synthesise_decoy(shape: Shape, members: Sequence[str]) -> str | None:
    """A value of this class's FORM that the class does not license, built from the members.

    The positive control has to be measurable even when nobody declared a negative — the ``url``
    class on a registry with no scan fact behind it is exactly that case, and treating the gate as
    ``not_measured`` there would deactivate the URL detector. Declared
    negatives are still preferred and still recorded as such: this is the floor, not the ceiling.

    Built by MUTATION rather than invention, so the decoy is guaranteed to have the form the rung
    selected. ``None`` when no mutation escaped the member set, which is reported as
    ``not_measured`` rather than assumed benign.
    """
    ordered = sorted((m for m in members if m.strip()), key=len, reverse=True)
    for base in ordered[:_DECOY_BASES]:
        for candidate in _decoy_candidates(base):
            if not shape.licenses(candidate, members) and shape.matches(candidate.strip()):
                return candidate
    return None


def _gate_positive_control(shape: Shape, members: Sequence[str], negatives: Sequence[str], prose: Sequence[str]) -> GateResult:
    """The detector must CATCH at least one non-member — and must read the text to do it.

    Two runs, because one is not enough. The first asks whether a negative handed to the detector
    directly is caught. The second injects that negative into a prose line and re-runs the lift: a
    detector whose with-decoy and without-decoy runs are identical never read the text at all,
    which is how a rule that lifts nothing passes every other gate.

    The control is the AUTHOR's declared negatives when the fact carries any and a synthesised
    one otherwise; which of the two decided is recorded, so the gate never claims the customer
    controlled a detector they said nothing about.
    """
    declared = [str(n) for n in negatives if str(n).strip()]
    control = declared or [d for d in (_synthesise_decoy(shape, members),) if d]
    origin = 'declared' if declared else 'synthesised'
    if not control:
        return GateResult(
            GATE_POSITIVE_CONTROL,
            None,
            {
                'negatives': len(declared),
                'reason': 'no negative was declared and none could be synthesised from the members, so nothing could control the detector',
            },
        )
    missed = [n for n in control if not any(not shape.licenses(v, members) for v in shape.lift(n))][:MAX_GATE_EVIDENCE]
    if missed:
        return GateResult(GATE_POSITIVE_CONTROL, False, {'negatives': len(declared), 'control': origin, 'missed': missed})
    carrier = prose[0] if prose else 'The team said'
    without = [v for v in shape.lift(carrier) if not shape.licenses(v, members)]
    with_decoy = [v for v in shape.lift(f'{carrier} {control[0]}') if not shape.licenses(v, members)]
    if with_decoy == without:
        return GateResult(
            GATE_POSITIVE_CONTROL,
            False,
            {
                'negatives': len(declared),
                'control': origin,
                'decoy': control[0],
                'reason': 'the with-decoy and without-decoy runs are identical, so the rule never read the text',
            },
        )
    return GateResult(GATE_POSITIVE_CONTROL, True, {'negatives': len(declared), 'control': origin, 'decoy': control[0]})


# =====================================================================================
# The ladder
# =====================================================================================


@dataclass(frozen=True)
class LadderResult:
    """The decision AND the rule object that made it.

    The :class:`Shape` is not serialisable and is deliberately not part of :class:`Detector`: the
    record is data a consumer reads back months later, and the callable is what THIS process uses
    to lift values out of a reply. :func:`detector_shape` re-derives the callable from a record.
    """

    detector: Detector
    shape: Shape | None


def measure_rung(
    rung: str, shape: Shape | None, members: Sequence[str], source: LadderInput, pattern: str | None = None, *, declared: bool = False
) -> tuple[Detector, Shape | None]:
    """Run all six gates against ONE candidate rung and record every result.

    PUBLIC, because "did gate G refuse rung R" has to be answerable directly: through the full
    descent a refused rung is invisible except as a line of prose, so a per-gate assertion would
    otherwise be a substring match on a message. ``derive_detector`` and the tests are the callers.

    Every gate is measured even after one fails, because the record is what a customer reads to
    decide what to change: "sample_floor refused" alone does not say whether adding members would
    also have to fix a specificity problem. The STATUS is decided by the first failure in
    :data:`GATES` order.
    """
    prose = _prose_haystack(source)
    unanimity = _gate_unanimity(shape, members, rung)
    if shape is None:
        return (
            Detector(
                rung=rung,
                status=STATUS_REFUSED,
                gates=(unanimity,),
                reason=unanimity.evidence.get('reason', 'no rule fits the members'),
                decided_by=source.decided_by,
            ),
            None,
        )
    gates = (
        unanimity,
        _gate_sample_floor(rung, members, source.families, declared=declared),
        _gate_specificity(shape, members, prose),
        _gate_script(shape, members, prose, source.reply_scripts),
        # The WHOLE source, record lines included — the haystack that makes this a different
        # question from specificity rather than the same sweep run twice.
        _gate_silence(shape, members, source.haystack),
        _gate_positive_control(shape, members, source.negatives, prose),
    )
    failed = next((g for g in gates if g.passed is not True), None)
    if failed is None:
        return (
            Detector(
                rung=rung,
                status=STATUS_ACTIVE,
                rule=shape.name,
                pattern=pattern,
                equivalence=shape.equivalence,
                decided_by=source.decided_by,
                gates=gates,
            ),
            shape,
        )
    return (
        Detector(
            rung=rung,
            status=STATUS_REFUSED if failed.passed is False else STATUS_NOT_MEASURED,
            rule='',
            pattern=pattern,
            equivalence=shape.equivalence,
            decided_by=source.decided_by,
            gates=gates,
            reason=failed.refusal(),
        ),
        None,
    )


def candidate_rungs(members: Sequence[str], *, override_pattern: str | None = None) -> list[tuple[str, Shape | None, str | None]]:
    """``(rung, rule-or-None, pattern-or-None)`` for each rung the ladder will try, in order.

    A rung whose rule is ``None`` had no candidate at all, which is a different failure from one
    whose candidate lost a gate — :func:`measure_rung` records both, and the report says which.
    An override REPLACES the descent rather than joining it: the customer has named the rule, and
    silently falling through to a rung they did not ask for would hide a refused declaration.
    """
    cleaned = [m.strip() for m in members if str(m).strip()]
    if override_pattern:
        return [(RUNG_INDUCED, _compiled_override(override_pattern), override_pattern)]
    induced = induce_pattern(cleaned)
    shape = matching_shape(cleaned)
    # THE CLASS'S OWN EQUIVALENCE IS NOT NEGOTIABLE. When a pre-validated shape fits the members
    # but its gates refused it, the ladder must NOT fall through to a rung that compares the same
    # values by a DIFFERENT rule: a refused ``url`` class descending to near-miss would compare
    # addresses by ``fold_key`` exactness, so ``http://www.docs.example.com/x`` stops being the
    # documented ``https://docs.example.com/x`` — the exoneration side goes dark while the judge
    # is still told the class is checked in code. Only a shape whose equivalence the later rungs
    # already share (``fold_key``) may fall through.
    keeps_its_own_rule = shape is not None and shape.equivalence != EQUIVALENCE_FOLD_KEY
    return [
        (RUNG_SHAPE, shape, None),
        (RUNG_INDUCED, None if keeps_its_own_rule else (_induced_shape(induced) if induced else None), induced),
        # The near-miss rung is REFUSED outright for members it cannot reason about, rather than
        # measured and failed: for a pure-letter member "one edit away" is the normal form of
        # correct prose, so every gate it could pass would be passing over the wrong question.
        (RUNG_NEAR_MISS, None if keeps_its_own_rule or not near_miss_admissible(cleaned) else _near_miss_shape(cleaned), None),
    ]


def derive_detector(members: Sequence[str], source: LadderInput | None = None, *, override_pattern: str | None = None) -> LadderResult:
    """Descend the ladder over ``members`` and return the first rung that passes all six gates.

    ``override_pattern`` is the customer's own declaration (``plurai.config.yaml``, or the
    ``detector.candidate.pattern`` their scan lifted from their code). It is VALIDATED, never
    trusted: it enters at rung 2 and runs the same six gates, so a pattern that fires on ordinary
    prose is refused with the offending text quoted. With no declared negatives the positive
    control still RUNS, against a decoy synthesised from the members — that proves the rule reads
    text, not that it catches what the customer cares about, and the record says which of the two
    controlled it.

    The returned detector's ``descent`` names every rung that was tried and why it did not
    activate, so a refusal a customer reads is a diagnosis rather than a verdict.
    """
    src = source or LadderInput()
    cleaned = [m.strip() for m in members if str(m).strip()]
    if not cleaned:
        return LadderResult(Detector(status=STATUS_REFUSED, reason='the class has no members to derive a detector from', decided_by=src.decided_by), None)

    descent: list[str] = []
    candidates = candidate_rungs(cleaned, override_pattern=override_pattern)

    declared = override_pattern is not None
    last = Detector(status=STATUS_REFUSED, reason='no rung was measured')
    for rung, shape, pattern in candidates:
        detector, active_shape = measure_rung(rung, shape, cleaned, src, pattern, declared=declared)
        if detector.active:
            return LadderResult(replace(detector, descent=tuple(descent)), active_shape)
        last = detector
        descent.append(f'{rung} → {detector.status}: {detector.reason}')

    # Every rung refused. The record keeps the LAST rung's gate measurements — the descent carries
    # the rest, so nothing about why is lost. Kept from the loop rather than re-measured: measuring
    # again is a second full sweep of the haystack for a record the loop already produced.
    return LadderResult(replace(last, rung=RUNG_NONE, reason=last.reason or 'no rung passed its gates', descent=tuple(descent)), None)


def closed_class_record(
    name: str,
    members: Sequence[str],
    *,
    subject: str = '',
    alias: str = '',
    source: str = '',
    total: int | None = None,
    ladder: LadderInput | None = None,
    override_pattern: str | None = None,
) -> dict[str, Any]:
    """ONE class record, wire shape and detector decision together.

    The single builder for the shape that rides ``source_grounding.closed_classes``,
    ``source_registry.yaml`` and the scan's ``closed_class_detectors`` section. Both sides call it
    — the backend over the registered units, the scan over the corpus roots — so a field added to
    the record cannot reach one side and miss the other, and a reader of either artifact is
    looking at the same keys.

    ``total`` defaults to the member count and is passed separately only where the two genuinely
    differ (the URL index publishes how many addresses the registry holds, which can exceed the
    members it carries on the wire).
    """
    deduped = list(dict.fromkeys(str(m) for m in members if str(m).strip()))
    detector = derive_detector(deduped, ladder, override_pattern=override_pattern).detector
    return {
        'class': name,
        'subject': subject or name.split(':', 1)[-1],
        'alias': alias or name,
        'members': deduped,
        'total': len(deduped) if total is None else int(total),
        'source': source,
        'detector': detector.as_dict(),
    }


class ClosedClassConfigError(ValueError):
    """A customer's declared closed-class detector is malformed. Fail closed, never "no override"."""


def _compiled_override(pattern: str) -> Shape:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ClosedClassConfigError(f'closed-class detector pattern {pattern!r} does not compile: {exc}') from exc
    return _induced_shape(pattern)


def detector_shape(detector: Detector, members: Sequence[str]) -> Shape | None:
    """Re-derive the LIFTING rule a recorded detector names, or ``None`` when it is not active.

    Reading the record rather than re-running the ladder is the whole point: a re-score
    months later must use the decision the registry carries, not whatever this build's ladder
    would decide today. Only the callable is rebuilt, from the recorded rung and pattern.
    """
    if not detector.active:
        return None
    if detector.rung == RUNG_SHAPE:
        return next((shape for shape in SHAPES if shape.name == detector.rule), None)
    if detector.rung == RUNG_INDUCED and detector.pattern:
        try:
            return _compiled_override(detector.pattern)
        except ClosedClassConfigError:
            return None
    if detector.rung == RUNG_NEAR_MISS:
        return _near_miss_shape([m for m in members if str(m).strip()])
    return None


# =====================================================================================
# Two-sided detection
# =====================================================================================


@dataclass(frozen=True)
class ClassDetection:
    """What the detector found in the texts it was pointed at — BOTH sides.

    The judge is handed the finding AND the exoneration. Reporting only the negative would give a
    correct citation no help avoiding a false red, and a "none outside the class" line over a
    reply that cited nothing would read exactly like one over a reply that cited five pages
    correctly.
    """

    #: Values the texts carry that nothing licenses.
    outside: tuple[str, ...] = ()
    #: Values the texts carry that the class's own MEMBERS license.
    licensed: tuple[str, ...] = ()
    #: Values licensed ONLY because a tool returned them — the world's values, not the class's.
    #: A separate bucket because collapsing it into :attr:`licensed` states something false: a
    #: deep link a mock invented is not in the customer's catalog, and telling a judge it is
    #: licensed BY THE LIST would launder a mock's invention into a documented address.
    tool_licensed: tuple[str, ...] = ()
    #: How many texts were read. Zero is a different fact from "nothing was found".
    scanned: int = 0


def detect(shape: Shape | None, members: Sequence[str], texts: Sequence[str], *, licensed_by: Sequence[str] = ()) -> ClassDetection:
    """Lift this class's values out of ``texts`` and split them by the class's own equivalence.

    ``licensed_by`` are TEXTS whose own values of the class additionally license it, read with the
    SAME rule rather than as a list of pre-extracted strings — so every class inherits the rule
    instead of the caller re-deciding it. It carries the run's TOOL RESULTS: a deep link a mock
    tool handed the agent is the WORLD's value, not the agent's invention, and pre-citing it as an
    outside-class value reds a faithful reply for repeating what its own tool returned.
    """
    if shape is None:
        return ClassDetection(scanned=len(texts))
    # The tool results are lifted ONCE per call. They are the whole run's, so this list grows with
    # the conversation while the reply under evaluation does not — and the judge's closed-class
    # block runs per policy per turn, so re-lifting them there would cost time that scales with
    # policies x turns.
    returned = [value for text in licensed_by for value in shape.lift(text)]
    outside: dict[str, None] = {}
    licensed: dict[str, None] = {}
    by_tool: dict[str, None] = {}
    for text in texts:
        for value in shape.lift(text):
            if shape.licenses(value, members):
                licensed.setdefault(value, None)
            elif returned and shape.licenses(value, returned):
                by_tool.setdefault(value, None)
            else:
                outside.setdefault(value, None)
    return ClassDetection(outside=tuple(outside), licensed=tuple(licensed), tool_licensed=tuple(by_tool), scanned=len(texts))


def near_miss_admissible(members: Sequence[str]) -> bool:
    """Whether the near-miss rung may run over these members at all: IDENTIFIERS ONLY.

    Every member must carry a digit, because a distance rule over pure-letter members reports
    ordinary correct prose as fabrication:

    * ``['Shipping', 'Billing', 'Onboarding']`` red ``shopping`` in "add the item to your
      shopping cart";
    * ``['Иванов', …]`` red ``Иванова``, the ordinary Russian genitive;
    * ``['Παπαδόπουλος', …]`` red ``Παπαδόπουλου``, the ordinary Greek genitive;
    * ``['Acme Video Platform', …]`` red ``Acme Video Platforms``, an English plural.

    The budget is one edit for any member under 16 key characters, and ONE EDIT IS THE NORMAL FORM
    of inflection in every language that inflects — a plural, a genitive, a definite article, a
    possessive. No amount of member length or script-listing separates ``Brandt``→``Brandtr`` from
    ``Platform``→``Platforms``: they are the same edit, and only a dictionary could tell them
    apart, which this module is forbidden to consult.

    A digit is the structural property that does separate them. ``1200043`` inflects in no
    language, so its one-edit neighbourhood contains no correct prose — only other identifiers,
    which is exactly the corruption the rung exists to catch (an id with one corrupted digit
    is caught, and the intact id is licensed). A pure-letter member is a WORD, and words
    have neighbours.

    The cost is a MISS on corrupted names and titles, which is the direction this whole module is
    biased toward: "A missing check costs a miss. A bad check invents a failure and destroys trust
    in the whole report." A Han, Thai or Hebrew word is pure-letter and is refused by the same
    test, so no hand-written list of boundaryless or agglutinating SCRIPTS is needed.
    """
    cleaned = [m for m in members if m.strip()]
    return bool(cleaned) and all(any(ch.isdigit() for ch in m) for m in cleaned)
