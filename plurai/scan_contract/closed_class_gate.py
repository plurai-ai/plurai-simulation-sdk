"""Does every closed set that REQUIRES a deterministic detector actually have one?

The gate that makes a missing check impossible to ship silently: a closed class either has a
deterministic detector or says that it has none.

**How the gate resolves "a detector exists for this variable" without hardcoding a name.** Not by
looking for a ``code_eval`` expression: a policy expression is authored per eval set, at scoring
time, and finding it would mean grepping for a helper name, which keys the gate on a name. It
runs the LADDER, over the members the fact's own subject
addresses, and reads the ladder's typed decision:

1. a ``closed_set`` fact with ``subject: leaders.name`` lifts its members from every corpus-unit
   field matching ``*.name`` — an fnmatch glob over the flattened ``CorpusUnit.fields`` paths.
   (Not ``facts._in_scope``, which globs UNIT IDS against a fact's ``scope``; the two ask
   different questions and share only the matcher.);
2. the ladder runs over those members, measured against the CLIENT-side haystack (every scanned
   unit's text), and the decision is recorded as a class record;
3. the gate is then "there exists a class whose ``subject`` is the fact's and whose detector is
   ``active``". No shape name, no helper name, no class name.

The two haystacks are deliberately not the same text — the client measures over the corpus roots,
the backend over the PRD's verbatim appendix plus staged materials plus knowledge-plan passages —
so a ``required`` detector active here and refused there is a pre-generation finding, the same
posture :func:`plurai.scan_contract.contract.reconcile_against_grounding` already takes.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from plurai.scan_contract.closed_classes import (
    DECIDED_BY_CLIENT,
    DECIDED_BY_OVERRIDE,
    STATUS_ACTIVE,
    ClosedClassConfigError,
    LadderInput,
    closed_class_record,
    line_key,
    read_detector,
)
from plurai.scan_contract.corpus import CorpusUnit
from plurai.scan_contract.render_actions import discover_config_documents
from plurai.scan_contract.facts import PREDICATE_CLOSED_SET, RUNG_INDUCED, ScanFact, declared_language_scripts

#: The artifact key carrying the class records this scan decided, so the backend can read the
#: customer's declarations off it and compare its own ladder run against the client's.
#:
#: DELIBERATELY the same string as :data:`CONFIG_KEY` below and as the ``report.py:CHECK_NAMES``
#: entry: the artifact section, the config key and the check are one feature, and a customer who
#: greps their config for the name should find the section it produced and the check that reported
#: it. Three constants rather than one because the three consumers must be able to move apart.
CLOSED_CLASSES_KEY = 'closed_class_detectors'

#: How many members one class carries into the record the report publishes. The same bound the
#: judge's licence already lives under, restated here because this side builds the record too.
GATE_MEMBER_LIMIT = 100


@dataclass(frozen=True)
class ClosedClassDecision:
    """One ``closed_set`` fact, the class its subject addressed, and what the ladder concluded.

    Carried as a record rather than as a boolean because the report has to say WHY: a refusal that
    names only "no detector" tells the customer nothing about whether to add members, supply a
    pattern, or accept judge-only checking.
    """

    fact: ScanFact
    #: The class record (:func:`closed_class_record` shape), or ``None`` when the subject
    #: addressed no corpus field at all — which is a different failure from a refused detector and
    #: must not be reported as one.
    record: Mapping[str, Any] | None
    #: Where the members were lifted from, for the report's provenance line.
    scopes: tuple[str, ...] = ()
    #: Set when the subject addressed MORE members than may ride the wire whole. A different
    #: failure from "no field matched" and from "the ladder refused", and it needs its own sentence:
    #: the remedy is to narrow the subject, not to add members or declare a pattern.
    oversized: int = 0

    @property
    def required(self) -> bool:
        return bool(self.fact.detector and self.fact.detector.required)

    @property
    def declared(self) -> bool:
        """Whether the author said anything at all. ``False`` is the third state — nobody said —
        which the report files as advisory rather than as a choice the customer made."""
        return self.fact.detector is not None

    @property
    def active(self) -> bool:
        return self.record is not None and read_detector(self.record.get('detector')).status == STATUS_ACTIVE

    def descent(self) -> str:
        """The ladder's whole descent, indented, or why there was nothing to descend."""
        if self.oversized:
            return (
                f'        the subject addressed {self.oversized} members, over the {GATE_MEMBER_LIMIT} a class may carry whole.\n'
                '        A truncated enumeration cannot be exhaustive — gating a rule against part of the list\n'
                '        produces one that reds the documented values it never saw. Narrow the subject to a column\n'
                '        that is genuinely enumerable, or drop the closed-set claim.'
            )
        if self.record is None:
            return (
                f'        no corpus field matched {self.fact.subject!r}, so there were no members to derive a detector\n'
                '        from. Check the subject spelling and the `source:` line — the members are lifted from the\n'
                "        cited unit, by the same scope rule the fact's own check uses."
            )
        detector = read_detector(self.record.get('detector'))
        lines = list(detector.descent) or [f'{detector.rung} → {detector.status}: {detector.reason}']
        return '\n'.join(f'        {line}' for line in lines)


#: What a customer can do about a refused requirement, stated once so every refusal says the same
#: three things. Deliberately ordered cheapest-first, and the third option is a real option.
GATE_REMEDY = (
    '        what to do: add members (the class may not be exhaustive), supply\n'
    '                    detector.candidate.pattern with its source line, or set\n'
    '                    detector.required: false and accept judge-only checking —\n'
    '                    which is stated in the report and in the judge prompt.'
)


#: The config key a customer declares detector overrides under, beside ``render_actions``. A
#: property of their domain, so it sits with their code.
CONFIG_KEY = 'closed_class_detectors'


def discover_closed_class_overrides(roots: Sequence[Path | str]) -> tuple[dict[str, str], list[str]]:
    """``({subject: pattern}, problems)`` declared under ``roots``.

    The same two rungs every workspace declaration is read at (:func:`discover_config_documents`):
    an absent config is silence. The shape is deliberately flat — a subject and the pattern that
    decides it, nothing else — because everything else about a detector is MEASURED, and a config
    key for a measurement would be an assertion the gates exist to refuse::

        closed_class_detectors:
          leaders.name: "^[A-Z][a-z]+ [A-Z][a-z]+$"
          orders.code: "^[A-Z]{2}-\\d{4}$"

    An override is VALIDATED, never trusted: it enters the ladder at rung 2 and runs the same six
    gates against the fact's negatives.

    ONE bad entry drops THAT entry, never the others, and is returned in ``problems`` naming its
    file and subject — never silently, because an override the customer wrote and this code
    dropped unsaid is the worst of the three outcomes: they believe a check runs and none does. A
    config that cannot be read, or a block that is not a mapping, drops what it declared. Two
    configs declaring the SAME subject differently drop that subject: the declaration is
    env-scoped, and a discovery order does not get to settle it.
    """
    declared: dict[str, tuple[str, Path]] = {}
    conflicted: set[str] = set()
    problems: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            documents = discover_config_documents(root)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for path, document in documents:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            raw = document.get(CONFIG_KEY)
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                problems.append(f'{path}: `{CONFIG_KEY}:` must be a mapping of subject -> regex, got {type(raw).__name__}.')
                continue
            for subject, pattern in raw.items():
                name = str(subject).strip()
                if not name or not str(pattern).strip():
                    problems.append(f'{path}: `{CONFIG_KEY}.{subject}` needs a non-blank subject and a non-blank regex.')
                    continue
                try:
                    # COMPILED HERE, where the file and the key can be named. The ladder compiles it
                    # too, but by then the error surfaces from inside a generation-time call with no
                    # idea which config wrote it.
                    re.compile(str(pattern))
                except re.error as exc:
                    problems.append(f'{path}: `{CONFIG_KEY}.{name}` is not a valid regex ({exc}): {pattern!r}.')
                    continue
                if name in declared and declared[name][0] != str(pattern):
                    problems.append(
                        f'conflicting `{CONFIG_KEY}` declarations for {name!r} ({declared[name][1]} and {path}); the declaration is '
                        'env-scoped and must agree. Reconcile the configs (or remove one).'
                    )
                    conflicted.add(name)
                    continue
                declared[name] = (str(pattern), path)
    return {name: pattern for name, (pattern, _) in declared.items() if name not in conflicted}, problems


def _subject_fields(subject: str) -> str:
    """The fnmatch glob a subject addresses: ``leaders.name`` -> ``*.name``.

    The COLUMN, across every record of the unit. A subject names ``<entity>.<field>`` and the
    flattened field paths carry the record index in between (``assets.3.link_url``), so an exact
    path would address one row and read its single value as the whole enumeration.
    """
    _, _, field_name = subject.rpartition('.')
    return f'*.{field_name}' if field_name else subject


def _candidate_pattern(fact: ScanFact) -> str | None:
    """The pattern the scan lifted from the customer's own code, when it lifted one.

    Only an ``induced`` candidate carries a pattern the ladder can run — a ``shape`` or
    ``near_miss`` candidate names a rung, and the rung is chosen by the members, not declared.
    """
    candidate = fact.detector.candidate if fact.detector else None
    return candidate.pattern if candidate and candidate.rung == RUNG_INDUCED and candidate.pattern.strip() else None


def _record_lines(unit: CorpusUnit, glob: str) -> list[str]:
    """The unit's lines that DECLARE a member of the subject's column, and only those.

    Matched THROUGH :func:`line_key` — the same reader the registry side parses records with, not
    a second one. Two constraints follow:

    * exclusion is by record shape, never by substring containment: a prose line that merely
      CONTAINS a member, such as a near-miss by suffix, is counter-evidence the gate must keep;
    * the record shape is the scanner's, not a split on the first colon: ``1. name: …`` and
      ``"name": …`` classify the same way on both sides, so one document yields one haystack and a
      class cannot be refused on one side and activated on the other.
    """
    keys = {path.rsplit('.', 1)[-1].casefold() for path in unit.fields if fnmatch.fnmatch(path, glob)}
    return [line for line in unit.text.splitlines() if line_key(line) in keys]


def _member_units(fact: ScanFact, units: Sequence[CorpusUnit], glob: str) -> list[CorpusUnit]:
    """The units the class's members are lifted from: those that CARRY the subject's column, with
    the fact's own citation used to disambiguate when it can.

    Two steps, in this order, and the order is the whole rule:

    1. keep only the units that actually carry a field matching the subject. A citation cannot
       conjure a column into a document that has none.
    2. if the ``source:`` names any of THOSE, narrow to them — the author pointed at the file the
       enumeration lives in and meant that file.

    Scoping to the citation FIRST is wrong: a ``closed_set`` fact is a claim about the data and
    routinely cites the machine-computed profile that counted the rows
    (``artifacts/data_profile.md:12``) rather than one of the files the rows live in. That
    citation resolves to a unit with no such column, and the class would come back with zero
    members, reported as "no corpus field matched": true of a scoping mistake, false of the
    customer's data.
    """
    carriers = [unit for unit in units if any(fnmatch.fnmatch(path, glob) for path in unit.fields)]
    cited = fact.source.split(':', 1)[0].strip()
    if not cited:
        return carriers
    named = [unit for unit in carriers if unit.unit_id == cited or unit.unit_id.endswith(f'/{cited}') or cited.endswith(unit.unit_id)]
    return named or carriers


def decide_closed_class_detectors(
    facts: Sequence[ScanFact],
    units: Sequence[CorpusUnit],
    *,
    overrides: Mapping[str, str] | None = None,
    reply_scripts: Sequence[str] = (),
) -> list[ClosedClassDecision]:
    """Run the ladder for every ``closed_set`` fact and record what it decided.

    ``overrides`` is the customer's ``plurai.config.yaml`` declaration, keyed by subject. It is
    VALIDATED, never trusted: it enters the ladder at rung 2 and runs the same six gates against
    the FACT's negatives, so a pattern that fires on ordinary prose is refused with the offending
    text quoted. A malformed one never arrives here: :func:`discover_closed_class_overrides` drops it
    and names it, and the shells file that as a report item.
    """
    declared = overrides or {}
    # Both of these depend only on ``units``, so they are built once, outside the per-fact loop:
    # the haystack is a full re-split of every unit, and the record-line scan re-splits them per
    # (fact, unit) pair. The record-line cache is keyed by the subject's GLOB because facts share
    # few of them.
    haystack = [line for unit in units for line in unit.text.splitlines() if line.strip()]
    record_lines: dict[tuple[str, str], list[str]] = {}
    decisions: list[ClosedClassDecision] = []
    for fact in facts:
        if fact.predicate != PREDICATE_CLOSED_SET:
            continue
        glob = _subject_fields(fact.subject)
        scoped = _member_units(fact, units, glob)
        members: list[str] = []
        scopes: list[str] = []
        member_lines: list[str] = []
        for unit in scoped:
            for path in sorted(unit.fields):
                if not fnmatch.fnmatch(path, glob):
                    continue
                value = unit.fields[path]
                if not str(value).strip():
                    continue
                members.append(str(value))
                scopes.append(f'{unit.unit_id}#{path}')
            key = (unit.unit_id, glob)
            if key not in record_lines:
                record_lines[key] = _record_lines(unit, glob)
            member_lines.extend(record_lines[key])
        if not members:
            decisions.append(ClosedClassDecision(fact=fact, record=None))
            continue
        if len(members) > GATE_MEMBER_LIMIT:
            # OVER THE CAP IS OPEN, never a truncated list gated as exhaustive — the invariant the
            # registry side already keeps. Gating a detector against the first 100 of 150 members
            # and publishing `total: 150` produces a rule that reds fifty documented values, with
            # a report saying every gate passed. A `required: true` fact over it is then fix-first,
            # which is the correct outcome: the class is too large to carry on the wire whole.
            decisions.append(ClosedClassDecision(fact=fact, record=None, oversized=len(members)))
            continue
        negatives = fact.detector.negatives if fact.detector else ()
        ladder = LadderInput(
            haystack=haystack,
            member_lines=member_lines,
            negatives=negatives,
            families=len({s.split('#', 1)[0] for s in scopes}),
            reply_scripts=reply_scripts,
            decided_by=DECIDED_BY_OVERRIDE if fact.subject in declared else DECIDED_BY_CLIENT,
        )
        record = closed_class_record(
            f'field:{fact.subject}',
            members[:GATE_MEMBER_LIMIT],
            subject=fact.subject,
            alias=f'field:{fact.subject.rpartition(".")[2] or fact.subject}',
            source=f'scan:{fact.source}' if fact.source else 'scan',
            total=len(members),
            ladder=ladder,
            # The config override wins; the fact's own `candidate.pattern` is the fallback. The
            # candidate runs all six gates like an override, and is still a PROPOSAL, so it can
            # still be refused.
            override_pattern=declared.get(fact.subject) or _candidate_pattern(fact),
        )
        # The author's NEGATIVES ride the record. The backend re-runs the positive control over
        # the registry it owns, and without them it can only control against a decoy it invents
        # for itself — which proves the rule reads text, not that it catches what the customer
        # said matters.
        decisions.append(ClosedClassDecision(fact=fact, record={**record, 'negatives': list(negatives)}, scopes=tuple(scopes)))
    return decisions


def declared_reply_scripts(language: str | None) -> tuple[str, ...]:
    """The writing systems an agent's declared reply language is spelled in.

    Fed to the script-agreement gate, which is the half a corpus census cannot answer on its own:
    an ``[A-Z]{2}`` induced over a Hebrew corpus fires on nothing in the sources, passes
    specificity vacuously, and then reds ``OK``, ``AI``, ``HR`` and ``PM`` in an English reply.

    The declaration is mapped as a LANGUAGE to its script (:func:`declared_language_scripts`),
    never read for the characters it happens to be spelled in: ``he``, ``Hebrew`` and ``ja`` are
    spelled in Latin letters but name non-Latin scripts. A language the mapping cannot place comes back as
    ``(SCRIPT_UNKNOWN,)``, which the gate reports as unmeasured rather than as a script.
    """
    return tuple(sorted(declared_language_scripts(language)))


__all__ = [
    'CLOSED_CLASSES_KEY',
    'CONFIG_KEY',
    'GATE_MEMBER_LIMIT',
    'GATE_REMEDY',
    'ClosedClassConfigError',
    'ClosedClassDecision',
    'decide_closed_class_detectors',
    'discover_closed_class_overrides',
    'declared_reply_scripts',
]
