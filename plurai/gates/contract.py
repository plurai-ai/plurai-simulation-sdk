"""The gate-authority wire types.

Every type here is frozen and refuses unknown fields: a finding or an overrule is a record that
binds by fingerprint, and a field silently dropped or added on one side is a binding the other
side cannot reproduce.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

#: Who may lift a gate's finding.
#:
#: * ``security`` — credentials, tenancy, billing, consent, privacy, isolation, no code execution from
#:   data. The operation stops; an overrule naming it is refused.
#: * ``invariant`` — downstream would crash or corrupt (malformed wire, unparseable schema, an
#:   overwrite of existing output). Not overrulable; the caller repairs or regenerates.
#: * ``advisory`` — everything else. The operation ships with the finding attached; a HOLDING
#:   advisory finding holds one named effect until a caller overrules it.
GateAuthority = Literal['security', 'invariant', 'advisory']
HARD_AUTHORITIES: Final[frozenset[str]] = frozenset({'security', 'invariant'})

#: What an overrule did, one per overrule a call carried.
#:
#: * ``applied`` — it matched a holding advisory finding by fingerprint and lifted that finding's hold.
#: * ``not_holding`` — it matched an advisory finding that holds nothing; nothing to lift.
#: * ``refused`` — it names a security or invariant gate, or a holding finding of missing evidence; no overrule lifts either.
#: * ``stale`` — its gate ran in this call and no current finding has its fingerprint: the evidence it
#:   was given for has changed (or never held).
#: * ``not_evaluated`` — its gate did not run in this call; it is neither applied nor stale.
OverruleOutcomeKind = Literal['applied', 'not_holding', 'refused', 'stale', 'not_evaluated']

#: The shortest overrule reason accepted, counted in characters after ``strip()`` (so any script counts the same).
OVERRULE_REASON_MIN_CHARS: Final[int] = 20

_FROZEN = ConfigDict(frozen=True, extra='forbid')

#: The canonicalisation :func:`fingerprint` applies. Bumped on any change to it: a client computing an
#: older canonical form produces fingerprints that never match, and the version says why.
FINGERPRINT_CANON_VERSION: Final[int] = 1


class EvidenceError(ValueError):
    """Evidence that cannot be fingerprinted stably: a float, a non-string key, a value JSON cannot carry,
    or a key set that is not exactly the gate's declared ``evidence_keys``."""


def canonical_evidence(value: Any, path: str) -> JsonValue:
    """``value`` with every string NFC-normalised, recursively. Refuses what has no stable canonical form."""
    if isinstance(value, str):
        return unicodedata.normalize('NFC', value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise EvidenceError(f'evidence at {path} is a float ({value!r}); encode a measurement as integers or a string so the fingerprint is stable.')
    if isinstance(value, Mapping):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError(f'evidence at {path} has a non-string key {key!r}.')
            normal = unicodedata.normalize('NFC', key)
            if normal in out:
                raise EvidenceError(f'evidence at {path} has two keys that normalise to {normal!r}.')
            out[normal] = canonical_evidence(item, f'{path}.{normal}')
        return out
    if isinstance(value, list | tuple):
        return [canonical_evidence(item, f'{path}[{i}]') for i, item in enumerate(value)]
    raise EvidenceError(f'evidence at {path} is a {type(value).__name__}, which JSON cannot carry.')


def fingerprint(gate_id: str, subject: str, evidence: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON of ``(gate_id, subject, evidence)``.

    Canonical form (version :data:`FINGERPRINT_CANON_VERSION`): every string NFC-normalised (keys and
    values, recursively); floats, NaN and non-string keys refused; tuples read as lists; ``json.dumps``
    with sorted keys, ``(',', ':')`` separators, ``ensure_ascii=False`` and ``allow_nan=False``, encoded
    UTF-8. Lists keep their order — a producer sorts an id list itself, since only it knows whether the
    order means something.
    """
    payload = {
        'v': FINGERPRINT_CANON_VERSION,
        'gate_id': canonical_evidence(gate_id, 'gate_id'),
        'subject': canonical_evidence(subject, 'subject'),
        'evidence': canonical_evidence(evidence, 'evidence'),
    }
    text = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class GateFinding(BaseModel):
    """What one gate saw about one subject. Build it with :func:`plurai.gates.apply.make_finding`.

    ``authority`` and ``holds`` are resolved from the catalogue, never chosen by the emitting site.
    ``fingerprint`` covers ``(gate_id, subject, evidence)`` only — never ``observed``, ``remedy``,
    ``draft_ref``, ``authority`` or ``holds`` — so an overrule keeps binding while those move.
    """

    model_config = _FROZEN

    gate_id: str
    authority: GateAuthority
    #: An advisory finding that holds ONE named effect (a release, a commit, an adoption) until overruled.
    #: Always False for security and invariant findings, which no overrule lifts.
    holds: bool
    #: What the finding is about: a policy, scenario, check, tool or set id.
    subject: str
    #: The stable, typed facts the gate decided on. Exactly the catalogue's ``evidence_keys``; no floats.
    evidence: dict[str, JsonValue]
    #: Volatile context (output tails, paths, timings, counts that move between runs). Never fingerprinted.
    observed: dict[str, JsonValue]
    #: What would make the gate pass.
    remedy: str
    #: The id of the work the gate held back, when it held some back; ``None`` otherwise.
    draft_ref: str | None
    fingerprint: str

    @model_validator(mode='after')
    def _canonical_by_construction(self) -> Self:
        """The invariants that need no catalogue, on every construction and every deserialisation: a hard finding
        holds nothing; the evidence is already in canonical form (NFC strings, no floats); the fingerprint is the
        one its ``(gate_id, subject, evidence)`` computes. Whether ``authority`` and ``holds`` are the CURRENT
        catalogue's is decided where the catalogue is in reach (``apply.make_finding`` / ``read_persisted_finding``)."""
        if self.authority in HARD_AUTHORITIES and self.holds:
            raise ValueError(f'{self.gate_id}: a {self.authority} finding never holds; holds must be False.')
        if canonical_evidence(self.evidence, 'evidence') != self.evidence:
            raise ValueError(f'{self.gate_id}: evidence is not in canonical form (every string NFC-normalised).')
        if self.fingerprint != fingerprint(self.gate_id, self.subject, self.evidence):
            raise ValueError(f'{self.gate_id}: fingerprint does not match the one (gate_id, subject, evidence) computes.')
        return self


class Overrule(BaseModel):
    """A caller lifting ONE finding: the gate id and the fingerprint of exactly the evidence it judged."""

    model_config = _FROZEN

    gate_id: str
    fingerprint: str
    #: Why the caller lifted it — what it verified. Internal: never rendered in a customer report.
    reason: str = Field(description=f'Why the finding is lifted; at least {OVERRULE_REASON_MIN_CHARS} characters.')

    @field_validator('reason')
    @classmethod
    def _reason_states_something(cls, value: str) -> str:
        if len(value.strip()) < OVERRULE_REASON_MIN_CHARS:
            raise ValueError(f'an overrule reason must state what was verified in at least {OVERRULE_REASON_MIN_CHARS} characters; got {len(value.strip())}.')
        return value

    @field_validator('gate_id', 'fingerprint')
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('gate_id and fingerprint must be non-empty.')
        return value


class OverruleOutcome(BaseModel):
    """What one overrule did. ``finding`` is the finding it matched, ``None`` when it matched none."""

    model_config = _FROZEN

    overrule: Overrule
    outcome: OverruleOutcomeKind
    finding: GateFinding | None

    @model_validator(mode='after')
    def _outcome_fits_finding(self) -> Self:
        """The outcome and the finding it names agree: a matched finding carries the overrule's binding; ``applied``
        lifted a HOLDING ADVISORY finding (never a security or invariant one); ``not_holding`` matched an advisory
        finding that holds nothing; ``stale`` matched none."""
        finding = self.finding
        if finding is not None and (finding.gate_id, finding.fingerprint) != (self.overrule.gate_id, self.overrule.fingerprint):
            raise ValueError("an overrule outcome names a finding whose (gate_id, fingerprint) is not the overrule's.")
        if self.outcome == 'applied' and (finding is None or finding.authority != 'advisory' or not finding.holds):
            raise ValueError('applied requires the holding advisory finding the overrule lifted.')
        if self.outcome == 'not_holding' and (finding is None or finding.authority != 'advisory' or finding.holds):
            raise ValueError('not_holding requires the advisory finding that holds nothing.')
        if self.outcome == 'stale' and finding is not None:
            raise ValueError('stale means no current finding carries the fingerprint; finding must be None.')
        return self


class GateAuthorityBlock(BaseModel):
    """The gate findings one operation produced and what each overrule it carried did — the one shape a
    job detail, a synchronous result and a client run record carry. Internal: never customer-rendered."""

    model_config = _FROZEN

    findings: list[GateFinding]
    overrules: list[OverruleOutcome]
