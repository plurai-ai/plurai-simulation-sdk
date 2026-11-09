"""Two consistency checks over the scan's OWN outputs — no corpus involved.

The rest of this package asks whether the scan's claims are true of the customer's documents.
These two ask whether the scan's outputs are consistent with EACH OTHER, which is the cheapest
class of pre-generation finding there is and the one nothing was running:

* :func:`check_agent_context_governance` — a suite can ship with ``agent_context`` UNGOVERNED on
  every scenario when no ``scenario_state_schema.yaml`` is supplied and nothing records that the
  agent is stateless. Ungoverned is not verified-empty: an agent that reads a current page at
  turn 0 sees an empty one in every scenario, and no artifact says so.
* :func:`check_complication_coverage` — the PRD can demand probe families that the typed
  complication pool carries no archetype for. Steering prose cannot deliver a family the
  batch-spread channel has no archetype for, and the two documents sit side by side at scan time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml
from plurai.scan_contract._safe_yaml import safe_yaml_load

#: The schema declares the fields and the scan supplied it.
STATE_GOVERNED = 'governed'
#: The schema was supplied and declares no ``agent_context`` field — an agent whose preconditions
#: really are empty, established rather than assumed.
STATE_VERIFIED_EMPTY = 'verified_empty'
#: No schema, and the scan recorded WHY the agent needs none.
STATE_STATELESS = 'stateless'
#: No schema and no finding. The precondition is unknown, and a generated scenario will carry
#: whatever the generator invents — or nothing.
STATE_UNGOVERNED = 'ungoverned'

#: Where the schema declares the agent's turn-0 precondition fields.
_AGENT_CONTEXT_KEY = 'agent_context_fields'

#: How the ``## Scenario design constraints`` section DECLARES its families (the PRD template's
#: spelling): one list item per family headed ``- Family <name>:`` (a dash also ends the label, and
#: a name may be quoted), or one ``Families:`` item or paragraph listing ``(A) …; (B) …``. Read as
#: markdown renders it: a soft line break is a space, and only the head of an item declares.
_BULLET = re.compile(r'^([ \t]*)[-*+][ \t]+')
_FAMILY_HEAD = re.compile(
    r'^[-*][ \t]+(?:\*\*)?Family[ \t]+(?:"(?P<quoted>[^"]{1,48})"|(?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,23}))(?:\*\*)?[ \t]*(?::|—|–|[ \t]-[ \t])',
    re.IGNORECASE,
)
#: A ``Families`` head: the item's leading text, emphasis stripped, is ``Families`` up to its first
#: ``:`` or ``.`` (``Families:``, ``**Families.**``, ``Families to cover, …:``).
_FAMILIES_HEAD = re.compile(r'^Families\b[^:.]*[:.](?P<body>.*)$', re.IGNORECASE)
_EMPHASIS = re.compile(r'\*+|__')
#: A listed family is a ``(X)`` that STARTS a segment: the list body, a ``;``-separated part of it,
#: or a DIRECT child item.
_LISTED_FAMILY = re.compile(r'^\((?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,23})\)')

#: The typed family link: each ``typical_complications`` archetype (or ``typical_user_profiles``
#: archetype, for a family defined by who the user is) lists, under this key, the declared family
#: labels it delivers. Written by the pools' author (extract-scenario-variables).
FAMILIES_KEY = 'families'

#: Why the complication-pool check did not run — typed, so a consumer can tell "the PRD declares
#: no family list" (the author's to fix) from "an input was absent".
NOT_RUN_NO_DIRECTIVES = 'no_directives'
NOT_RUN_NO_POOL = 'no_pool'
NOT_RUN_FAMILIES_UNDECLARED = 'families_undeclared'
NOT_RUN_LINKS_UNDECLARED = 'links_undeclared'


@dataclass(frozen=True)
class StateGovernance:
    """Whether ``agent_context`` is governed for this agent, and by what."""

    status: str
    detail: str
    fields: tuple[str, ...] = ()

    @property
    def decided(self) -> bool:
        """Whether the scan established an answer. ``False`` for :data:`STATE_UNGOVERNED` — which
        is the whole point: an ungoverned precondition must render as a check that did not run,
        never as one that ran and found nothing to say."""
        return self.status != STATE_UNGOVERNED

    def as_dict(self) -> dict[str, Any]:
        return {'status': self.status, 'detail': self.detail, 'fields': list(self.fields)}


def check_agent_context_governance(schema_yaml: str | None, stateless_finding: str = '') -> StateGovernance:
    """Decide ``agent_context`` governance from the state schema, or from an explicit finding.

    ``schema_yaml`` is ``scenario_state_schema.yaml``'s text when the scan has one. It legitimately
    does not exist before the mock env is built, which is why the second input exists: a scan that
    established the agent reads no turn-0 state records that here, and the contract then reports a
    DECIDED check instead of a blocked one. Neither supplied is :data:`STATE_UNGOVERNED` — not an
    error, but not a pass either.
    """
    if schema_yaml is not None and schema_yaml.strip():
        try:
            document = safe_yaml_load(schema_yaml)
        except yaml.YAMLError as exc:
            return StateGovernance(
                status=STATE_UNGOVERNED,
                detail=f'the supplied scenario_state_schema.yaml does not parse ({" ".join(str(exc).split())[:160]}), so no field could be read out of it',
            )
        # THREE different facts, and collapsing them would report "ungoverned" as verified-empty.
        # A comments-only file parses to ``None``; a bare list or a scalar parses to something with
        # no keys; a key nested one level too deep is simply absent. None of those may fall into the
        # arm that says "the schema declares the precondition empty" — an assertion about a
        # document that says nothing of the sort.
        if not isinstance(document, Mapping):
            return StateGovernance(
                status=STATE_UNGOVERNED,
                detail=f'the supplied scenario_state_schema.yaml parsed as {type(document).__name__}, not a mapping, so '
                f'`{_AGENT_CONTEXT_KEY}` could not be looked up in it at all — this is a file that was READ, not a schema '
                'that was consulted',
            )
        if _AGENT_CONTEXT_KEY not in document:
            top_level = ', '.join(str(k) for k in list(document)[:8]) or '(none)'
            return StateGovernance(
                status=STATE_UNGOVERNED,
                detail=f'the supplied scenario_state_schema.yaml carries no `{_AGENT_CONTEXT_KEY}` key at all (its top-level '
                f'keys are: {top_level}). ABSENT is not EMPTY: a key nested one level too deep, or spelled differently, reads '
                f'exactly like an agent with no turn-0 precondition. Declare `{_AGENT_CONTEXT_KEY}: []` to say the '
                'precondition really is empty',
            )
        declared = document[_AGENT_CONTEXT_KEY]
        if declared is not None and not isinstance(declared, (list, tuple, Mapping)):
            return StateGovernance(
                status=STATE_UNGOVERNED,
                detail=f'`{_AGENT_CONTEXT_KEY}` in the supplied scenario_state_schema.yaml is a {type(declared).__name__} '
                f'({declared!r}), not a list or mapping of field names, so no field could be read out of it — and a value '
                'nothing can read is not an empty precondition',
            )
        names = tuple(str(name) for name in declared) if declared is not None else ()
        if names:
            return StateGovernance(status=STATE_GOVERNED, detail=f'scenario_state_schema.yaml declares {len(names)} agent_context field(s)', fields=names)
        return StateGovernance(
            status=STATE_VERIFIED_EMPTY,
            detail=f'scenario_state_schema.yaml was supplied and its `{_AGENT_CONTEXT_KEY}` is present and empty — the '
            'precondition is empty because the schema says so, not by omission',
        )
    if stateless_finding.strip():
        return StateGovernance(status=STATE_STATELESS, detail=f'no state schema, and the scan recorded why: {stateless_finding.strip()}')
    return StateGovernance(
        status=STATE_UNGOVERNED,
        detail='no scenario_state_schema.yaml and no stateless finding, so `agent_context` is UNGOVERNED — not verified-empty. '
        'Every generated scenario opens with an empty precondition and no artifact says whether that is correct; if this agent '
        'reads a current page, selected record or active account at turn 0 it will see an EMPTY one in every scenario. Supply '
        'the schema, or record what the scan established about the agent being stateless',
    )


@dataclass(frozen=True)
class FamilyCoverage:
    """One scenario family the PRD's directives name, and the pool archetypes that carry it."""

    family: str
    archetypes: tuple[str, ...] = ()

    @property
    def linked(self) -> bool:
        return bool(self.archetypes)


@dataclass(frozen=True)
class ComplicationCoverage:
    """Whether the typed complication pool can deliver the families the PRD demands."""

    checked: bool
    blocked_reason: str = ''
    families: tuple[FamilyCoverage, ...] = ()
    pool: tuple[str, ...] = ()
    #: One of the ``NOT_RUN_*`` kinds when ``checked`` is False.
    not_run: str = ''
    #: ``(label, archetypes)`` for every ``families:`` label that names no declared family — a
    #: misspelled or split link, refused rather than dropped.
    unknown_links: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: The unknown labels written as ``(label)`` in the section: present, but not at the start of a
    #: list entry — a list whose entries are not separated by ``;``.
    unseparated: tuple[str, ...] = ()

    @property
    def unlinked(self) -> tuple[FamilyCoverage, ...]:
        return tuple(f for f in self.families if not f.linked)

    def summary(self) -> str:
        if not self.checked:
            return f'complication-pool coverage NOT checked: {self.blocked_reason}'
        unlinked = self.unlinked
        if not unlinked:
            return f'all {len(self.families)} declared family(ies) have an archetype listing them under `{FAMILIES_KEY}:`'
        return (
            f'{len(unlinked)} of {len(self.families)} declared family(ies) have no complication or user-profile archetype listing '
            f'them under `{FAMILIES_KEY}:`: {", ".join(f.family for f in unlinked)}. Those pools are what the backend spreads '
            'across a batch, so a family no archetype carries can only be asked for in steering prose'
        )


def _pool_names(variables: Mapping[str, Any] | None) -> tuple[str, ...]:
    entries = (variables or {}).get('typical_complications')
    return tuple(str(name) for entry in entries if isinstance(entry, Mapping) and (name := entry.get('name'))) if isinstance(entries, list) else ()


#: The indent a heading block carries: below every paragraph (-1) and item (>= 0), so it ends any list.
_HEADING = -2


@dataclass(frozen=True)
class _Item:
    """One block of the section: a list item (its bullet's indent), a paragraph (indent ``-1``, or its
    own indent when indented under an item), or a heading (``_HEADING``)."""

    indent: int
    #: The block's text with each soft line break read as a space.
    text: str
    is_bullet: bool = False


def _items(directives: str) -> list[_Item]:
    """The section's blocks, in order. A blank line ends a block; a heading is a block of its own."""
    blocks: list[_Item] = []
    open_block = False
    for line in directives.splitlines():
        stripped = line.strip()
        if not stripped:
            open_block = False
        elif stripped.startswith('#'):
            blocks.append(_Item(_HEADING, ''))
            open_block = False
        elif bullet := _BULLET.match(line):
            blocks.append(_Item(len(bullet.group(1).expandtabs(4)), line[bullet.end() :].strip(), is_bullet=True))
            open_block = True
        elif open_block:
            blocks[-1] = _Item(blocks[-1].indent, f'{blocks[-1].text} {stripped}', blocks[-1].is_bullet)
        else:
            lead = line[: len(line) - len(line.lstrip())]
            blocks.append(_Item(len(lead.expandtabs(4)) if lead else -1, stripped))
            open_block = True
    return blocks


def _declared_families(directives: str) -> list[str]:
    """The family labels the section DECLARES, in order and de-duplicated. Empty when none is declared.

    A ``Families`` list is the head's own body split on ``;`` — an intro ending in ``:`` before the
    first ``(X)`` is skipped — plus each DIRECT child item, split the same way. A grandchild, a wrapped
    line, an indented paragraph inside a child or a following sibling starts nothing; a heading ends the list.
    """
    declared: dict[str, None] = {}
    items = _items(directives)
    for index, item in enumerate(items):
        if item.is_bullet and (head := _FAMILY_HEAD.match('- ' + item.text)):
            declared.setdefault((head.group('quoted') or head.group('name')).strip())
            continue
        if item.indent == _HEADING or (not item.is_bullet and item.indent > -1):
            continue  # a heading, or a paragraph indented under an item, declares nothing
        listing = _FAMILIES_HEAD.match(_EMPHASIS.sub('', item.text).strip())
        if listing is None:
            continue
        segments = listing.group('body').split(';')
        if not _LISTED_FAMILY.match(segments[0].strip()) and ':' in segments[0]:
            segments[0] = segments[0].split(':', 1)[1]
        depth: int | None = None
        for child in items[index + 1 :]:
            if child.indent <= item.indent:
                break
            if not child.is_bullet:
                continue
            depth = child.indent if depth is None else depth
            if child.indent == depth:
                segments.extend(_EMPHASIS.sub('', child.text).split(';'))
        for segment in segments:
            if listed := _LISTED_FAMILY.match(segment.strip()):
                declared.setdefault(listed.group('name'))
    return list(declared)


def _family_links(variables: Mapping[str, Any] | None) -> dict[str, list[str]] | None:
    """``families:`` label, as written → the archetypes that list it. ``None`` when no archetype declares any."""
    links: dict[str, list[str]] = {}
    declared_any = False
    for pool, label_key in (('typical_complications', 'name'), ('typical_user_profiles', 'style')):
        entries = (variables or {}).get(pool)
        for entry in entries if isinstance(entries, list) else ():
            labels = entry.get(FAMILIES_KEY) if isinstance(entry, Mapping) else None
            if not isinstance(labels, list) or not entry.get(label_key):
                continue
            declared_any = True
            for label in labels:
                links.setdefault(str(label).strip(), []).append(str(entry[label_key]))
    return links if declared_any else None


def check_complication_coverage(directives: str | None, variables: Mapping[str, Any] | None, variables_reason: str = '') -> ComplicationCoverage:
    """Cross-reference the PRD's demanded scenario families against the typed complication pool.

    ``directives`` is the routed ``## Scenario design constraints`` span, whose declared family
    labels are the demanded side. A family is COVERED when a pool archetype lists its label under
    ``families:`` — a typed link, so coverage is set arithmetic, never a reading of prose. The check
    reports the whole table as one finding rather than one per family, because the author acts on it.

    Never blocking. The link between a family and an archetype is an authoring choice, and a
    family can legitimately be shaped by an opening rather than by a complication — so this
    reports what the two documents say about each other and leaves the judgement to the reader.
    """
    pool = _pool_names(variables)
    if not directives or not directives.strip():
        return ComplicationCoverage(
            checked=False,
            not_run=NOT_RUN_NO_DIRECTIVES,
            blocked_reason='no `## Scenario design constraints` directives reached this check, so there is no demanded family to compare the pool against',
            pool=pool,
        )
    if not pool:
        # The REASON, when the loader has one. "No pool reached this check" over a file that is
        # sitting right there, one bad indent from parsing, sends the author to the plumbing.
        return ComplicationCoverage(
            checked=False,
            not_run=NOT_RUN_NO_POOL,
            blocked_reason='no `typical_complications` pool reached this check, so nothing could be compared against the demanded families'
            + (f' — {variables_reason}' if variables_reason else ''),
        )
    families = _declared_families(directives)
    if not families:
        return ComplicationCoverage(
            checked=False,
            not_run=NOT_RUN_FAMILIES_UNDECLARED,
            blocked_reason='the `## Scenario design constraints` section declares no family — declare them as `- Families: (A) …; (B) …` '
            'or one `- Family <name>: …` entry each',
            pool=pool,
        )
    links = _family_links(variables)
    if links is None:
        return ComplicationCoverage(
            checked=False,
            not_run=NOT_RUN_LINKS_UNDECLARED,
            blocked_reason=f'no complication or user-profile archetype records which declared family it delivers (`{FAMILIES_KEY}: [<label>, …]`)',
            # The declared side is still reported: it was read, only the link to compare it with is missing.
            families=tuple(FamilyCoverage(family=family) for family in families),
            pool=pool,
        )
    # Labels compare case-insensitively; one written in two cases links both ways.
    by_key: dict[str, list[str]] = {}
    for label, names in links.items():
        by_key.setdefault(label.casefold(), []).extend(names)
    covered = [FamilyCoverage(family=family, archetypes=tuple(by_key.get(family.casefold(), ()))) for family in families]
    declared = {family.casefold() for family in families}
    unknown = tuple((label, tuple(names)) for label, names in links.items() if label.casefold() not in declared)
    unseparated = tuple(label for label, _ in unknown if re.search(rf'\(\s*{re.escape(label)}\s*\)', directives, re.IGNORECASE))
    return ComplicationCoverage(checked=True, families=tuple(covered), pool=pool, unknown_links=unknown, unseparated=unseparated)


def load_variables(text: str | None) -> tuple[Mapping[str, Any] | None, str]:
    """Parse a ``scenario_variables.yaml``, and say WHY when it could not be read.

    Never raises: this runs beside checks a malformed input must not take down. Returns the
    document and an empty reason, or ``None`` and the reason.

    The reason is the whole second element. Swallowing the ``YAMLError`` would make a malformed
    file report as "no `typical_complications` pool reached this check" — a sentence about
    plumbing, which sends the author to the script that passed the file rather than to the line
    in their file that does not parse. The reason embeds the parse error, in the same voice as the
    sibling schema check.
    """
    if not text or not text.strip():
        return None, 'no scenario_variables.yaml was supplied'
    try:
        document = safe_yaml_load(text)
    except yaml.YAMLError as exc:
        return None, f'the supplied scenario_variables.yaml does not parse ({" ".join(str(exc).split())[:160]})'
    if not isinstance(document, Mapping):
        return None, f'the supplied scenario_variables.yaml parsed as {type(document).__name__}, not a mapping'
    return document, ''
