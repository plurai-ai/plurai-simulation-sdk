"""Pre-upload disclosure preview + append-only audit log.

The SDK is the egress chokepoint, so it discloses what it is about to send and records
what it sent to ``.plurai/uploads/audit.log``. **Disclosure and audit run loudly in BOTH
postures**: an ``anonymized`` env previews the scrub summary + key
fingerprint and records them; a ``raw`` env prints a loud ``RAW (anonymization off)``
banner and records ``posture: raw`` with no fingerprint. Either way, when a customer asks
"what exactly did you receive?", the per-upload posture record answers it. Because
ingestion is a Bash invocation, the normal Bash-approval prompt is the consent chokepoint;
this module makes the decision informed and auditable.

The audit log is also the **per-workspace mixed-posture fence**
(:func:`assert_posture_consistent`): before a run egresses, it refuses to add
a differently-postured upload to an env that already shipped one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plurai import _private_files
from plurai._platform import is_windows
from plurai.ingest.anonymize import KEY_DIR_NAME, _ensure_key_dir_gitignored
from plurai.ingest.folder_names import PSEUDONYMIZED_FOLDER_KINDS
from plurai.ingest.posture import CONFIG_FILENAME, POSTURE_CONFIG_KEY, AnonymizationPosture

logger = logging.getLogger(__name__)

#: Where the Terms of Service and the Privacy Policy that govern an upload are published. Each is printed in the
#: disclosure when set. An empty one prints nothing.
TERMS_OF_SERVICE_URL = ''
PRIVACY_POLICY_URL = ''

#: Append-only audit log location, relative to the workspace root.
AUDIT_LOG_RELPATH = f'{KEY_DIR_NAME}/uploads/audit.log'
#: Where the last disclosure an ingest command showed is kept, beside the audit log. One file per delivery kind, so
#: deliveries of different kinds that run at the same time do not overwrite each other's file.
DISCLOSURE_RELPATH_TEMPLATE = f'{KEY_DIR_NAME}/uploads/disclosure-{{kind}}.txt'
#: The audit record ``event`` of a part's upload outcome and of a part whose upload failed.
PART_UPLOAD_EVENT = 'part_upload'
PART_UPLOAD_FAILED_EVENT = 'part_upload_failed'


def _legal_lines() -> list[str]:
    """One line per published legal document the upload is covered by."""
    return [
        *([f'  terms of service: {TERMS_OF_SERVICE_URL}'] if TERMS_OF_SERVICE_URL else []),
        *([f'  privacy policy:   {PRIVACY_POLICY_URL}'] if PRIVACY_POLICY_URL else []),
    ]


def disclosure_relpath(kind: str) -> str:
    """The disclosure file of a delivery kind, relative to the workspace root."""
    return DISCLOSURE_RELPATH_TEMPLATE.format(kind=''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in kind))


class IngestAuditError(Exception):
    """The archive ALREADY egressed (uploaded → server-minted lease) but the local
    compliance audit record could not be written. Egress-aware: it is raised only AFTER the upload
    succeeds, so it never means "nothing left the machine" — it means "something left but was not
    recorded locally". The message names the lease id + checksum + byte count so the operator can
    record the egress by hand. Caught by the CLI ``main()`` handler → clean exit, not a raw traceback."""


class MixedPostureError(ValueError):
    """Raised (fail-closed, BEFORE egress) when a run's posture disagrees with an upload this
    workspace already made. A raw↔pseudonym mix within one backend env breaks grounding joins
    *silently* (a scenario references ``ACC_hash`` while the corpus holds raw ``ACC_1234``), so the
    env is refused rather than allowed to ingest artifacts under two postures. Caught by the CLI
    ``main()`` handler → clean exit, not a raw traceback."""


def assert_posture_consistent(workspace_root: str | Path, posture: AnonymizationPosture) -> None:
    """Fail closed (:class:`MixedPostureError`) if this workspace already uploaded under a *different*
    posture than ``posture``. No audit log yet → pass (nothing to conflict with).

    A record with **no** ``posture`` field counts as ``'anonymized'``: the field is optional in the
    log format, and a record without it is read as anonymized. Unparseable audit
    lines are skipped with a loud ``logger.warning`` rather than aborting the check.

    **Honesty note.** This fence is per-workspace / per-machine — the audit log is local and
    gitignored, so it cannot see uploads a *second* machine made into the same backend env. That is
    exactly why every upload also pins its posture onto the wire (``manifest.posture``): the
    cross-machine enforcement is the backend finalize/build refusing a mixed-posture env, and this
    check is the local first line of defense.
    """
    log_path = Path(workspace_root) / AUDIT_LOG_RELPATH
    if not log_path.exists():
        return  # no prior uploads in this workspace → nothing to conflict with
    with log_path.open('r', encoding='utf-8') as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning('skipping an unparseable %s line during the posture-consistency check: %r', log_path, stripped[:80])
                continue
            if record.get('planned'):
                continue  # a plan sent nothing, so it fixes no posture for the env
            if record.get('event') in (PART_UPLOAD_EVENT, PART_UPLOAD_FAILED_EVENT):
                continue  # an upload outcome restates a part record that already fixed the posture
            if 'posture' in record and record['posture'] is None:
                continue  # an agent description part is not scrubbed and has no posture to conflict with
            prior = record.get('posture', 'anonymized')  # a record without the field is read as anonymized
            if prior != posture:
                raise MixedPostureError(
                    f'this env already uploaded under posture {prior!r} '
                    f'(audit record kind={record.get("kind")!r} at {record.get("timestamp")!r}), '
                    f'but this run resolved posture {posture!r} from {CONFIG_FILENAME}. '
                    'A mixed-posture env mixes raw values with pseudonyms and breaks grounding joins, so it is refused. '
                    f'Remedy: restore the previous {POSTURE_CONFIG_KEY!r} value in {CONFIG_FILENAME}, or start a fresh env — '
                    f're-ingest ALL artifacts under the new posture into a new backend env and clear {log_path}.'
                )


@dataclass(frozen=True)
class UploadPreview:
    """The deterministic pre-upload disclosure shown before bytes leave the machine.

    ``scrub_summary`` / ``key_fingerprint`` are present iff ``posture == 'anonymized'`` (a raw run
    scrubs nothing and mints no key), enforced by :meth:`__post_init__`.

    ``posture_source`` is the config file the posture was READ FROM
    (:attr:`~plurai.ingest.posture.PostureResolution.source`), or ``None`` for the default. It is
    the difference between "someone declared ``anonymize: false`` here" and "nobody said
    anything", so the raw banner only nags an operator to opt in when nobody has declined
    already."""

    kind: str  # the ingest pipeline: 'corpus' | 'traces' | 'doc-examples' | 'scenario-bundle' | 'run-artifacts' | 'run-report'
    unit_count: int
    total_bytes: int
    target: str
    posture: AnonymizationPosture
    scrub_summary: dict[str, int] | None = None
    key_fingerprint: str | None = None
    posture_source: Path | None = None
    # Where the units were read from when that is a local folder rather than a live sink, e.g. ``export_dir /path``.
    source: str | None = None

    def __post_init__(self) -> None:
        if self.posture == 'anonymized' and (self.scrub_summary is None or self.key_fingerprint is None):
            raise ValueError('anonymized UploadPreview requires both scrub_summary and key_fingerprint')
        if self.posture == 'raw' and (self.scrub_summary is not None or self.key_fingerprint is not None):
            raise ValueError('raw UploadPreview must carry neither scrub_summary nor key_fingerprint (no scrub runs under raw)')

    def _source_lines(self) -> list[str]:
        return [f'  source:  {self.source}'] if self.source else []

    def render(self) -> str:
        if self.posture == 'raw':
            # LOUD banner: raw egress is a deliberate, DPA-governed choice — say so unmissably.
            # The PROVENANCE line separates a declined posture from an unset one. Both egress
            # raw, so both keep the unscrubbed warning; only the second is news, and only the
            # second gets the opt-in nag — repeating it at an operator who already wrote
            # ``anonymize: false`` teaches them to skim the whole banner.
            declared = self.posture_source is not None
            return '\n'.join(
                [
                    'Plurai ingestion — pre-upload disclosure',
                    f'  posture: RAW (declared in {self.posture_source})' if declared else '  posture: RAW (anonymization off — nothing declared one)',
                    '  !! data will be uploaded UNSCRUBBED; processing is governed by your DPA with Plurai',
                    *([] if declared else ["  !! opt in per env: set 'anonymize: true' in plurai.config.yaml (requires plurai-simulation-sdk[anonymize])"]),
                    f'  kind:    {self.kind}',
                    f'  target:  {self.target}',
                    *self._source_lines(),
                    f'  units:   {self.unit_count} (payload files bundled, not source documents)',
                    f'  bytes:   {self.total_bytes} (raw)',
                    *_legal_lines(),
                ]
            )
        scrub_summary = self.scrub_summary or {}
        scrub = f'{scrub_summary.get("freetext_values", 0)} free-text value(s)'
        lines = [
            'Plurai ingestion — pre-upload disclosure',
            f'  kind:    {self.kind}',
            f'  target:  {self.target}',
            *self._source_lines(),
            f'  units:   {self.unit_count}',
            f'  bytes:   {self.total_bytes} (anonymized)',
            f'  scrubbed: {scrub}',
            f'  key fingerprint: {self.key_fingerprint}',
            f'  posture: anonymized (declared in {self.posture_source})' if self.posture_source is not None else '  posture: anonymized',
            (
                '  note: scenario folder names are replaced by keyed pseudonyms (`plurai folder-names` reads them back), '
                'except the minted shapes: regression_<slug>, thread-<id>, scenario_<n>. A regression slug is sent as its '
                'recorder wrote it. The paths inside each folder are structural addressing and are NOT anonymized.'
                if self.kind in PSEUDONYMIZED_FOLDER_KINDS
                else '  note: manifest keys / object names are structural addressing and are NOT anonymized.'
            ),
            *_legal_lines(),
        ]
        return '\n'.join(lines)


def write_audit_entry(
    workspace_root: str | Path,
    *,
    kind: str,
    lease_id: str,
    checksum: str,
    archive_bytes: int,
    member_keys: list[str],
    posture: AnonymizationPosture,
    member_sources: dict[str, str] | None = None,
    key_fingerprint: str | None = None,
    scrub_summary: dict[str, int] | None = None,
) -> Path:
    """Append ONE JSONL record for the single uploaded archive to ``.plurai/uploads/audit.log``.

    Under the single-archive transport the whole run egresses as ONE compressed
    object → ONE lease, so the audit is ONE record per run (append-only: a later run appends another
    line). The record ALWAYS carries the archive's lease ref, SHA-256 (``checksum``), byte count, the
    bundled member KEYS (WHICH files were sent — the disclosure trail), and the run's ``posture``
    (recorded in BOTH postures). The key ``fingerprint`` + ``scrub`` summary are
    included ONLY under ``anonymized`` posture (a raw run mints no key and scrubs nothing); never the
    pseudonym key, and never any content. ``member_sources`` maps a member name the run pseudonymized
    to the local unit key it came from; it is recorded when given, and it never leaves the machine,
    so it is how an operator finds the local file behind a pseudonymous name. Returns the audit-log path.

    ``.plurai/`` is made self-ignoring here (not only in ``load_or_create_key``) so a raw-posture run —
    which never mints a key — still cannot leave the audit log trackable in git.
    """
    log_path = Path(workspace_root) / AUDIT_LOG_RELPATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Raw posture never calls load_or_create_key, so ensure the .plurai/.gitignore exists here too —
    # the audit log carries upload metadata that must never be committed.
    _ensure_key_dir_gitignored(log_path.parent.parent)
    try:
        log_path.parent.chmod(0o700)
        if is_windows():
            _private_files.restrict_directory(log_path.parent)
    except OSError as e:  # pragma: no cover - best effort on exotic filesystems
        logger.warning('could not restrict %s to 0700 (%s); audit metadata dir may be broadly readable', log_path.parent, e)
    record = {
        'timestamp': datetime.now(UTC).isoformat(),
        'kind': kind,
        'posture': posture,
        'archive': {'lease_id': lease_id, 'checksum': checksum, 'bytes': archive_bytes},
        'members': list(member_keys),  # WHICH files were bundled (addressing keys) — the disclosure trail
    }
    if posture == 'anonymized':
        # Only an anonymized run has a key fingerprint + scrub summary to record.
        record['key_fingerprint'] = key_fingerprint
        record['scrub'] = scrub_summary
    if member_sources is not None:
        record['member_sources'] = dict(member_sources)
    try:
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    except OSError as e:
        # The bytes already egressed (this is called AFTER upload succeeds) — a failed audit write
        # must not be a bare traceback that hides WHAT left. Surface the lease/checksum/bytes so the
        # operator can record the egress by hand.
        raise IngestAuditError(
            f'archive already egressed as lease {lease_id} (checksum {checksum}, {archive_bytes} bytes) '
            f'but the audit record could not be written to {log_path}: {e}. Record this egress manually.'
        ) from e
    return log_path


def _uploads_dir(workspace_root: str | Path) -> Path:
    uploads = Path(workspace_root) / AUDIT_LOG_RELPATH
    uploads.parent.mkdir(parents=True, exist_ok=True)
    _ensure_key_dir_gitignored(uploads.parent.parent)
    try:
        uploads.parent.chmod(0o700)
        if is_windows():
            _private_files.restrict_directory(uploads.parent)
    except OSError as e:  # pragma: no cover - best effort on exotic filesystems
        logger.warning('could not restrict %s to 0700 (%s); audit metadata dir may be broadly readable', uploads.parent, e)
    return uploads.parent


def write_disclosure_file(workspace_root: str | Path, kind: str, text: str) -> Path:
    """Keep the disclosure an ingest command printed, so what the operator was shown outlives the terminal."""
    path = _uploads_dir(workspace_root) / Path(disclosure_relpath(kind)).name
    path.write_text(text + '\n', encoding='utf-8')
    return path


def write_part_audit_entries(workspace_root: str | Path, *, kind: str, records: list[dict[str, Any]]) -> Path:
    """Append ONE JSONL record per part to ``.plurai/uploads/audit.log``, BEFORE any of its bytes leave.

    Each record is the caller's ``part`` description (kind, name or label, ``sha256``, ``bytes``) plus
    ``posture`` (``None`` for an agent description, which is sent as written), the key fingerprint and scrub
    summary of an anonymized part, the archive's member names and the pseudonymised-name map. The record is
    written when the part is disclosed: a part the backend already holds is recorded too, since it is part
    of what the operator approved. Never the pseudonym key, and never any content.
    """
    log_path = _uploads_dir(workspace_root) / Path(AUDIT_LOG_RELPATH).name
    stamp = datetime.now(UTC).isoformat()
    try:
        with log_path.open('a', encoding='utf-8') as fh:
            for record in records:
                fh.write(json.dumps({'timestamp': stamp, 'kind': kind, **record}, ensure_ascii=False) + '\n')
    except OSError as e:
        raise IngestAuditError(f'the audit record could not be written to {log_path}: {e}. Nothing was uploaded.') from e
    return log_path


def write_part_outcome_entry(workspace_root: str | Path, *, kind: str, record: dict[str, Any]) -> Path:
    """Append ONE JSONL record to ``.plurai/uploads/audit.log`` after a part's upload ended.

    ``record`` is the outcome: ``event`` :data:`PART_UPLOAD_EVENT` with the part's ``sha256``, whether this run
    sent it (``uploaded``) and the ``object_id`` that holds it, or :data:`PART_UPLOAD_FAILED_EVENT` with the part
    that failed. The part's disclosure record was written before the upload; this one says what became of it.
    """
    log_path = _uploads_dir(workspace_root) / Path(AUDIT_LOG_RELPATH).name
    try:
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps({'timestamp': datetime.now(UTC).isoformat(), 'kind': kind, **record}, ensure_ascii=False) + '\n')
    except OSError as e:
        raise IngestAuditError(
            f'the upload outcome of a part ({json.dumps(record, ensure_ascii=False)}) could not be written to {log_path}: {e}. Record it manually.'
        ) from e
    return log_path


__all__ = [
    'AUDIT_LOG_RELPATH',
    'DISCLOSURE_RELPATH_TEMPLATE',
    'PART_UPLOAD_EVENT',
    'PART_UPLOAD_FAILED_EVENT',
    'IngestAuditError',
    'MixedPostureError',
    'UploadPreview',
    'assert_posture_consistent',
    'disclosure_relpath',
    'write_audit_entry',
    'write_disclosure_file',
    'write_part_audit_entries',
    'write_part_outcome_entry',
]
