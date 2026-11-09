"""Content-addressed parts: the printed part entry, and the disclose-then-upload gate.

A part is one piece of a workspace's grounding package, addressed by the SHA-256 of its bytes. An ingest
command builds its parts locally, discloses every one with its size and posture, records one audit entry
per part, and only then sends the bytes the backend does not hold. The entry it prints is what the plugin
passes to ``ground_workspace_start``; it names the part and carries no manifest::

    {"kind": "trace_archive", "sha256": "sha256:<hex>", "bytes": 5211044, "posture": "anonymized", "anon_key_fingerprint": "<hex>"}

"Part" here is the package's unit. The storage routes' own ``/parts/{i}`` chunks are upload chunks.

A plan builds the same parts and stops there: :func:`plan_parts` records them as planned and nothing is sent.
A run that carries the hashes of an approved plan compares them to the parts it built before it discloses or
sends anything (:func:`require_expected_sha256`, :func:`require_expected_sha256_by_name`).

The gate has no way around the disclosure: :func:`upload_disclosed` takes the :class:`Disclosure` that
:func:`disclose_parts` returns, uploads exactly the parts it lists, and re-hashes each file first, so a
part that changes or appears after the disclosure aborts the run instead of leaving the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from plurai.ingest.disclosure import (
    IngestAuditError,
    PART_UPLOAD_EVENT,
    PART_UPLOAD_FAILED_EVENT,
    disclosure_relpath,
    write_disclosure_file,
    write_part_audit_entries,
    write_part_outcome_entry,
)
from plurai.ingest.posture import AnonymizationPosture
from plurai.ingest.storage_client import PartUpload, StorageClient

PartKind = Literal['agent_description', 'trace_archive', 'corpus_source']

#: The agent descriptions the backend reads; any other description is stored and not read.
DESCRIPTION_FILES: Final[tuple[str, ...]] = ('plurai.config.yaml', 'simulation_prd.md', 'scenario_variables.yaml')
#: The scan's claims about the agent, delivered as a description; the plugin writes it outside the description folder.
SCAN_CONTRACT_NAME: Final[str] = 'scan_contract.json'

logger = logging.getLogger(__name__)

_SHA256 = re.compile(r'^sha256:[0-9a-f]{64}$')
_HASH_CHUNK_BYTES: Final[int] = 1024 * 1024


class PartSetChangedError(Exception):
    """A part to upload is not the one that was disclosed: its bytes changed, or it was never listed."""


class PartMismatchError(Exception):
    """A part built now is not the one an approved plan named. Raised before anything is disclosed or sent."""


class PartEntry(BaseModel):
    """The printed entry for one part: kind, name or label, hash, size, posture and fingerprint, nothing else."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: PartKind
    name: str | None = None
    label: str | None = None
    sha256: str
    bytes: int = Field(ge=0)
    posture: AnonymizationPosture | None = None
    anon_key_fingerprint: str | None = None

    @model_validator(mode='after')
    def _shape(self) -> PartEntry:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError('sha256 must be "sha256:" and 64 lowercase hex characters')
        if self.kind == 'agent_description':
            if not self.name:
                raise ValueError('an agent description part needs a name')
            if self.posture is not None or self.anon_key_fingerprint is not None:
                raise ValueError('an agent description part carries no posture or fingerprint')
            return self
        if self.posture is None:
            raise ValueError('an evidence part states its posture')
        if self.kind == 'corpus_source' and not self.label:
            raise ValueError('a corpus_source part needs a label')
        if self.posture == 'anonymized' and not self.anon_key_fingerprint:
            raise ValueError('an anonymized part carries the fingerprint of the key it was scrubbed under')
        if self.posture == 'raw' and self.anon_key_fingerprint is not None:
            raise ValueError('a raw part carries no fingerprint')
        return self

    @model_serializer(mode='wrap')
    def _omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}

    @property
    def title(self) -> str:
        return f'{self.kind} {self.name or self.label or ""}'.rstrip()


@dataclass(frozen=True)
class BuiltPart:
    """A part ready to send: its entry, the file holding exactly the bytes the entry names, and how to store it."""

    entry: PartEntry
    path: Path
    object_name: str
    content_type: str
    members: tuple[str, ...] = ()
    verbatim: tuple[str, ...] = ()
    member_sources: dict[str, str] | None = None
    scrub_summary: dict[str, int] | None = None


@dataclass(frozen=True)
class Disclosure:
    """The parts the operator was shown, with an audit entry written for each. The only way to upload.

    ``workspace_root`` and ``kind`` say where the upload outcome of each part is recorded.
    """

    parts: tuple[BuiltPart, ...]
    workspace_root: str | Path
    kind: str


def sha256_of(path: Path) -> tuple[str, int]:
    """``("sha256:<hex>", size)`` of a file, read in bounded chunks."""
    hasher = hashlib.sha256()
    size = 0
    with path.open('rb') as handle:
        while block := handle.read(_HASH_CHUNK_BYTES):
            hasher.update(block)
            size += len(block)
    return f'sha256:{hasher.hexdigest()}', size


def part_lines(parts: tuple[BuiltPart, ...] | list[BuiltPart]) -> list[str]:
    """One line per part: what it is, its size, its posture and its hash."""
    lines = [f'  parts:   {len(parts)}']
    for part in parts:
        entry = part.entry
        posture = entry.posture if entry.posture is not None else 'none (agent description, sent as written)'
        lines.append(f'    - {entry.title}: {entry.bytes} bytes, posture {posture}, {entry.sha256}')
        if part.verbatim:
            lines.append(f'      sent as written, not scrubbed: {", ".join(part.verbatim)}')
    return lines


def render_parts_disclosure(parts: tuple[BuiltPart, ...] | list[BuiltPart], *, target: str) -> str:
    """The disclosure for parts that go out as written (agent descriptions), which have no scrub to report."""
    return '\n'.join(['Plurai ingestion — pre-upload disclosure', f'  target:  {target}', *part_lines(parts)])


def _audit_record(part: BuiltPart) -> dict[str, Any]:
    entry = part.entry
    record: dict[str, Any] = {
        'posture': entry.posture,
        'part': {key: value for key, value in entry.model_dump(mode='json').items() if key in ('kind', 'name', 'label', 'sha256', 'bytes')},
    }
    if part.members:
        record['members'] = list(part.members)
    if part.verbatim:
        record['verbatim'] = list(part.verbatim)
    if entry.posture == 'anonymized':
        record['key_fingerprint'] = entry.anon_key_fingerprint
        record['scrub'] = part.scrub_summary
    if part.member_sources is not None:
        record['member_sources'] = dict(part.member_sources)
    return record


def planned_entry(entry: PartEntry) -> dict[str, Any]:
    """The JSON a plan prints for a part: the entry a real run prints, marked ``planned``."""
    return {**entry.model_dump(mode='json'), 'planned': True}


def plan_parts(parts: Sequence[BuiltPart], *, workspace_root: str | Path, kind: str) -> None:
    """Record one ``planned`` audit entry per part. Nothing is disclosed, so nothing is sent or shown as an upload."""
    write_part_audit_entries(workspace_root, kind=kind, records=[{**_audit_record(part), 'planned': True} for part in parts])


def sha256_argument(raw: str) -> str:
    """An ``--expect-sha256`` value for a single part: ``sha256:<64 lowercase hex>``."""
    if not _SHA256.fullmatch(raw):
        raise argparse.ArgumentTypeError(f'{raw!r} is not "sha256:" and 64 lowercase hex characters')
    return raw


def named_sha256_argument(raw: str) -> tuple[str, str]:
    """An ``--expect-sha256`` value for one of several parts: ``<name>=sha256:<64 lowercase hex>``."""
    name, separator, digest = raw.rpartition('=')
    if not separator or not name:
        raise argparse.ArgumentTypeError(f'{raw!r} is not <part name>=sha256:<hex>')
    return name, sha256_argument(digest)


def require_expected_sha256(entry: PartEntry, expected: str) -> None:
    """Raise :class:`PartMismatchError` unless the part built is the one the plan named."""
    if entry.sha256 != expected:
        raise PartMismatchError(f'{entry.title} is {entry.sha256} but the approved plan named {expected}; nothing was uploaded')


def require_expected_sha256_by_name(entries: Sequence[PartEntry], expected: Mapping[str, str]) -> None:
    """Raise :class:`PartMismatchError` unless the parts built are exactly the ones the plan named, matched by name.

    An entry nobody expected and an expectation no entry meets are both mismatches; the message states every one.
    """
    problems: list[str] = []
    built = {entry.name: entry for entry in entries}
    for name, entry in built.items():
        if name not in expected:
            problems.append(f'{entry.title} is {entry.sha256} but the approved plan named nothing for it')
        elif entry.sha256 != expected[name]:
            problems.append(f'{entry.title} is {entry.sha256} but the approved plan named {expected[name]}')
    problems += [f'agent_description {name} was approved as {digest} but is no longer a part' for name, digest in expected.items() if name not in built]
    if problems:
        raise PartMismatchError('; '.join(problems) + '; nothing was uploaded')


def disclose_parts(
    parts: list[BuiltPart],
    *,
    workspace_root: str | Path,
    kind: str,
    text: str,
    err: TextIO,
) -> Disclosure:
    """Show every part, write the disclosure file and one audit entry per part. Nothing has been sent yet."""
    print(text, file=err)
    write_disclosure_file(workspace_root, kind, text)
    write_part_audit_entries(workspace_root, kind=kind, records=[_audit_record(part) for part in parts])
    return Disclosure(parts=tuple(parts), workspace_root=workspace_root, kind=kind)


async def upload_disclosed(client: StorageClient, disclosure: Disclosure) -> dict[str, PartUpload]:
    """Send the disclosed parts the backend lacks, keyed by ``sha256``.

    Each file is re-hashed before anything is sent: a part whose bytes differ from its entry raises
    :class:`PartSetChangedError`, and nothing is uploaded. A backend without the by-hash lookup raises
    :class:`~plurai.ingest.storage_client.StorageUploadError` before it is sent anything.

    Each part's outcome is appended to the audit log as soon as its upload ends: ``uploaded`` and ``object_id``
    when it succeeds, and a record naming the part when it fails, so the log shows how far a run got.
    """
    for part in disclosure.parts:
        if sha256_of(part.path) != (part.entry.sha256, part.entry.bytes):
            raise PartSetChangedError(f'{part.entry.title} changed after it was disclosed; nothing was uploaded')
    uploaded: dict[str, PartUpload] = {}
    for part in disclosure.parts:
        entry = part.entry
        try:
            outcome = await client.upload_if_absent(
                part.path,
                name=part.object_name,
                checksum=entry.sha256.removeprefix('sha256:'),
                content_length=entry.bytes,
                content_type=part.content_type,
            )
        except Exception as failure:
            _record_failed_part(disclosure, part, failure)
            raise
        uploaded[entry.sha256] = outcome
        write_part_outcome_entry(
            disclosure.workspace_root,
            kind=disclosure.kind,
            record={'event': PART_UPLOAD_EVENT, 'sha256': entry.sha256, 'uploaded': outcome.uploaded, 'object_id': outcome.object_id},
        )
    return uploaded


def _record_failed_part(disclosure: Disclosure, part: BuiltPart, failure: Exception) -> None:
    """Append the record of the part whose upload failed. The failure itself is the error the caller sees, so a record
    that cannot be written is logged and does not replace it."""
    entry = part.entry
    try:
        write_part_outcome_entry(
            disclosure.workspace_root,
            kind=disclosure.kind,
            record={'event': PART_UPLOAD_FAILED_EVENT, 'sha256': entry.sha256, 'part': entry.title, 'error': type(failure).__name__},
        )
    except IngestAuditError as unwritten:
        logger.warning('%s', unwritten)


__all__ = [
    'DESCRIPTION_FILES',
    'BuiltPart',
    'Disclosure',
    'PartEntry',
    'PartKind',
    'PartMismatchError',
    'PartSetChangedError',
    'SCAN_CONTRACT_NAME',
    'disclose_parts',
    'disclosure_relpath',
    'named_sha256_argument',
    'part_lines',
    'plan_parts',
    'planned_entry',
    'require_expected_sha256',
    'require_expected_sha256_by_name',
    'render_parts_disclosure',
    'sha256_argument',
    'sha256_of',
    'upload_disclosed',
]
