"""Which scan claims a corpus part's scrub hid, for the scrub map the part carries.

The scan decides claims about the agent against its corpus (``scan_contract.json``, ``fact_verification``). Each decided
claim is a ledger key, ``agent_fact:<subject>.fact:<predicate>``, and its check matches literal terms in the corpus text.
When the scrub rewrites text holding one of those terms, the part no longer shows what the scan saw, so the backend must
read the claim as unseen. This module reads the claims and reports, after a unit is scrubbed, the claims whose terms
the scrub removed. A claim with no terms cannot be tied to any text and is never listed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from plurai.ingest.anonymize import Anonymizer
from plurai.ingest.scrub_map import ScrubRecorder, build_key


@dataclass(frozen=True)
class Claim:
    """A decided scan claim: its ledger key and the terms its check matches on, as :func:`normalized` text."""

    key: str
    terms: tuple[str, ...]


class _Unit(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def path(self) -> Path: ...


def normalized(text: str) -> str:
    """Whitespace collapsed and case folded: the text the scan's check matches its terms against."""
    return ' '.join(text.split()).casefold()


def _predicate(row: Mapping[str, Any]) -> str:
    """The predicate a verdict row declares: its own field, else the suffix of its fact id after the subject."""
    if predicate := str(row.get('predicate') or '').strip():
        return predicate
    fact_id, subject = (str(row.get(name) or '').strip() for name in ('fact_id', 'subject'))
    return fact_id[len(subject) + 1 :] if subject and fact_id.startswith(f'{subject}.') else ''


def _terms(row: Mapping[str, Any]) -> tuple[str, ...]:
    check = row.get('check')
    raw = (check if isinstance(check, Mapping) else row).get('terms')
    return tuple(term for item in raw if isinstance(item, str) and (term := normalized(item))) if isinstance(raw, list) else ()


def load_claims(contract: Mapping[str, Any]) -> tuple[Claim, ...]:
    """The claims ``contract`` decided (verified or contradicted) that name a key and hold terms to look for."""
    from plurai.scan_contract.facts import CONTRADICTED, VERIFIED, artifact_fact_verdicts

    claims: list[Claim] = []
    for row in artifact_fact_verdicts(contract) or ():
        if not isinstance(row, Mapping) or row.get('status') not in (VERIFIED, CONTRADICTED):
            continue
        subject = str(row.get('subject') or '').strip()
        key = build_key('agent_fact', subject, ('fact', _predicate(row))) if subject else None
        if key is not None and (terms := _terms(row)):
            claims.append(Claim(key=key, terms=terms))
    return tuple(claims)


def _leaves(value: Any) -> Iterator[str]:  # noqa: ANN401 - decoded JSON
    if isinstance(value, str):
        yield normalized(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)


def _counts(path: Path, terms: frozenset[str]) -> dict[str, int] | None:
    """How often each term occurs across the string values of the JSON file at ``path``, or ``None`` when it is not JSON.

    Each string value is counted on its own, so a term that spans two values (a chunk boundary) is not counted.
    """
    try:
        leaves = list(_leaves(json.loads(path.read_text(encoding='utf-8'))))
    except (ValueError, UnicodeDecodeError):
        return None
    return {term: sum(leaf.count(term) for leaf in leaves) for term in terms}


def claim_scrub_unit[UnitT: _Unit](
    inner: Callable[[Anonymizer, UnitT], None], claims: tuple[Claim, ...], recorder: ScrubRecorder
) -> Callable[[Anonymizer, UnitT], None]:
    """``inner`` with a record, in ``recorder``, of each claim whose terms the scrub removed from the unit.

    A term the scrub left fewer times than it found marks its claim: the scrub rewrote text the claim's check matched.
    A unit that is not JSON is scrubbed by ``inner`` alone.
    """
    terms = frozenset(term for claim in claims for term in claim.terms)

    def scrub(scrubber: Anonymizer, unit: UnitT) -> None:
        before = _counts(unit.path, terms)
        inner(scrubber, unit)
        after = _counts(unit.path, terms) if before is not None else None
        if before is None or after is None:
            return
        for claim in claims:
            if any(after[term] < before[term] for term in claim.terms):
                recorder.add(claim.key)

    return scrub
