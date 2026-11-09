"""The scan contract's gate ids — shared by every catalogue that resolves them.

The checker runs in two shells (the backend's CLI here, ``plurai-scan-contract`` in the
SDK, which vendors this package), and both mint and overrule the same findings, so the ids and the
evidence each one fingerprints are declared ONCE, here, and each side's catalogue builds its row
from :data:`SCAN_GATES` with its own ``check`` command as the carrier.

Every scan gate is ADVISORY and HOLDS one effect: the scan's ``proceed`` signal, i.e. whether
generation starts on this scan's premises. An overrule releases that hold and nothing else — a
contradicted fact stays contradicted on the artifact, and only a VERIFIED fact ever feeds a probe, a
pool entry or a world constraint.

Pure data: no import of any other module, so a catalogue may import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

#: A typed fact the corpus refutes (absence matched, existence missing, count off). Evidence names the
#: check and the scopes that decided it, so an overrule lapses when either moves.
#: The namespace every scan gate id lives in. A scan overrule is selected by it, never by membership
#: in :data:`SCAN_GATES`: a misspelled ``scan.*`` id must reach the catalogue and be refused there,
#: not be filtered out here and lift nothing while the run reads as though it had been judged.
SCAN_GATE_PREFIX: Final[str] = 'scan.'

FACT_CONTRADICTED: Final[str] = 'scan.fact_contradicted'
#: A ``closed_set`` fact whose author declared ``detector.required: true`` and no detector passed.
CLOSED_CLASS_DETECTOR_MISSING: Final[str] = 'scan.closed_class_detector_missing'
#: A ``--corpus`` root that admitted no file at the scan's suffix scope.
CORPUS_ROOT_EMPTY: Final[str] = 'scan.corpus_root_empty'


@dataclass(frozen=True)
class ScanGate:
    """What both catalogues need to build a scan gate's row: why it holds, and what it fingerprints."""

    why: str
    evidence_keys: tuple[str, ...]


#: THE scan gates. Read-only. Every one is advisory and holding (see the module docstring).
SCAN_GATES: Final[Mapping[str, ScanGate]] = MappingProxyType(
    {
        FACT_CONTRADICTED: ScanGate(
            why='the PRD still states the refuted claim to every generator, so generation waits until the fact is re-authored or the claim is judged to stand',
            evidence_keys=('fact_id', 'kind', 'check', 'scopes_addressed', 'matches'),
        ),
        CLOSED_CLASS_DETECTOR_MISSING: ScanGate(
            why='the author required a code-side detector for this closed set and none passed; generation waits for one or for an explicit judge-only decision',
            evidence_keys=('fact_id', 'subject', 'member_count', 'oversized', 'descent'),
        ),
        CORPUS_ROOT_EMPTY: ScanGate(
            why='a corpus root that admits no file decides nothing about the facts scoped to it; generation waits for a wider scope or the root dropped',
            evidence_keys=('root', 'admitted_suffixes', 'skipped_by_suffix'),
        ),
    }
)
