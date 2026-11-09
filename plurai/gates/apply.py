"""The pure gate-authority functions that need the catalogue: build a finding, re-read a persisted one, apply overrules."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, cast

from pydantic import JsonValue

from plurai.gates.catalogue import GateSpec, gate_spec
from plurai.gates.contract import HARD_AUTHORITIES, EvidenceError, GateFinding, Overrule, OverruleOutcome, canonical_evidence, fingerprint


def make_finding(
    gate_id: str,
    *,
    subject: str,
    evidence: Mapping[str, Any],
    observed: Mapping[str, Any],
    remedy: str,
    draft_ref: str | None,
) -> GateFinding:
    """THE way to build a :class:`GateFinding`.

    ``gate_id`` must be a catalogue constant: an undeclared id raises
    :class:`~plurai.gates.catalogue.UnknownGateError`. ``evidence`` must carry exactly the gate's
    declared ``evidence_keys`` (:class:`EvidenceError` otherwise). ``authority`` and ``holds`` come from
    the catalogue and cannot be passed. Every argument is required; ``draft_ref`` is an explicit ``None``
    when the gate held no work back.
    """
    spec = gate_spec(gate_id)
    keys = set(evidence)
    if keys != spec.evidence_keys:
        raise EvidenceError(
            f'{gate_id}: evidence keys {sorted(keys)} differ from the declared {sorted(spec.evidence_keys)} '
            f'(missing {sorted(spec.evidence_keys - keys)}, unexpected {sorted(keys - spec.evidence_keys)}).'
        )
    return GateFinding(
        gate_id=gate_id,
        authority=spec.authority,
        holds=spec.holds,
        subject=subject,
        evidence=cast('dict[str, JsonValue]', canonical_evidence(evidence, 'evidence')),
        observed=cast('dict[str, JsonValue]', _json_safe(observed)),
        remedy=remedy,
        draft_ref=draft_ref,
        fingerprint=fingerprint(gate_id, subject, evidence),
    )


def read_persisted_finding(row: Any) -> GateFinding | None:
    """A finding an earlier job persisted, re-read under the CURRENT catalogue; ``None`` when it no longer fits.

    ``authority`` and ``holds`` are re-stamped from the catalogue, never kept from the row: a gate reclassified
    since the row was written reads back as it is classified today. The row's fingerprint must equal the one
    its ``(gate_id, subject, evidence)`` computes today (:class:`GateFinding` checks it, with no floats and NFC
    evidence).

    ``None`` for a row that is not a :class:`GateFinding`, names a gate the catalogue no longer declares,
    carries an evidence key set other than the gate's current ``evidence_keys`` (the gate's evidence grew or
    shrank since the row was written), or carries a fingerprint the recomputation does not reproduce. Such a
    row is stale: its fingerprint can never match one minted today, so keeping it would leave a finding no
    overrule answers. The caller drops it with a warning and the slot's next authoring mints the current shape.
    Never raises.
    """
    if isinstance(row, GateFinding):
        row = row.model_dump(mode='json')
    if not isinstance(row, Mapping) or not isinstance(gate_id := row.get('gate_id'), str):
        return None
    try:
        spec = gate_spec(gate_id)
        finding = _stamped(row, spec)
    except ValueError:
        return None
    return finding if set(finding.evidence) == spec.evidence_keys else None


def _stamped(row: Mapping[str, Any], spec: GateSpec) -> GateFinding:
    """``row`` as a :class:`GateFinding` carrying the catalogue's ``authority`` and ``holds``, validated (so its
    fingerprint is re-checked). Raises ``ValueError`` on a row that does not validate."""
    return GateFinding.model_validate({**row, 'authority': spec.authority, 'holds': spec.holds})


def _json_safe(value: Any) -> JsonValue:
    """``observed`` is never fingerprinted, so a float (a timing) is allowed there; a non-finite float and
    any value JSON cannot carry are rendered as text rather than refused, since context must not cost a finding."""
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | bool) or value is None:
        return value
    return str(value)


#: The ``observed['kind']`` of a finding that states missing evidence. The hold lasts until new evidence arrives, so no overrule lifts it.
UNLIFTABLE_KINDS: Final[frozenset[str]] = frozenset({'evidence_missing'})


def lifts_hold(finding: GateFinding) -> bool:
    """True when an overrule can lift ``finding``: it holds, and what it states is not missing evidence."""
    return finding.holds and finding.observed.get('kind') not in UNLIFTABLE_KINDS


def apply_overrules(
    findings: Sequence[GateFinding],
    overrules: Sequence[Overrule],
    *,
    evaluated_gates: Collection[str],
) -> list[OverruleOutcome]:
    """One :class:`OverruleOutcome` per overrule, in order. Pure: never changes a finding.

    ``evaluated_gates`` is every gate that RAN in this call, whether or not it produced a finding; it is
    what separates ``stale`` (the gate ran and nothing carries this fingerprint) from ``not_evaluated``
    (the gate did not run here, so the overrule is neither applied nor lapsed).

    The outcome is decided from the CATALOGUE's ``authority`` and ``holds``, never from the matched finding's.
    An overrule naming a security or invariant gate is ``refused`` whatever it matches, and so is one that matches a
    holding finding of missing evidence (:data:`UNLIFTABLE_KINDS`). An overrule whose
    gate id is not in the catalogue raises :class:`~plurai.gates.catalogue.UnknownGateError` — a typo
    must not read as a gate that merely did not run.
    """
    evaluated = frozenset(evaluated_gates)
    by_key = {(f.gate_id, f.fingerprint): f for f in findings}
    out: list[OverruleOutcome] = []
    for overrule in overrules:
        spec = gate_spec(overrule.gate_id)
        match = by_key.get((overrule.gate_id, overrule.fingerprint))
        if match is not None and (match.authority, match.holds) != (spec.authority, spec.holds):
            # A finding stamped under an earlier classification of its gate: the outcome names it as it reads today.
            match = _stamped(match.model_dump(mode='json'), spec)
        if spec.authority in HARD_AUTHORITIES:
            outcome = 'refused'
        elif overrule.gate_id not in evaluated:
            outcome = 'not_evaluated'
        elif match is None:
            outcome = 'stale'
        elif not spec.holds:
            outcome = 'not_holding'
        elif match.observed.get('kind') in UNLIFTABLE_KINDS:
            outcome = 'refused'
        else:
            outcome = 'applied'
        out.append(OverruleOutcome(overrule=overrule, outcome=outcome, finding=match))
    return out


def applied_fingerprints(outcomes: Sequence[OverruleOutcome]) -> frozenset[str]:
    """The fingerprints of every finding an ``applied`` outcome lifted."""
    return frozenset(o.finding.fingerprint for o in outcomes if o.outcome == 'applied' and o.finding is not None)
