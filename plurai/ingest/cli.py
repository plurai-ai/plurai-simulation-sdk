"""``plurai ingest-*`` CLI: extract → (anonymize) → bundle → manifest.

Each subcommand runs the existing local extractor, applies the env's **anonymization posture**
— under ``anonymized`` it scrubs the output with the full scrubber (fail-closed — no raw path
*within* an anonymized env's build uploads); under ``raw`` (the default) it bundles the extractor output
verbatim, minting no key and never importing Presidio — then it bundles ALL units into ONE streamed
``.tar.gz`` (tar member name == each unit's manifest ``key``), uploads that archive ONCE to the storage service via
:class:`~plurai.ingest.storage_client.StorageClient`, writes the disclosure preview +
``.plurai/uploads/audit.log``, and prints the **single-archive manifest** to stdout so the plugin
can hand it to the backend's ``*_start(manifest=...)`` MCP tool (the server mints the job/set id).
The run kinds (``ingest-run``, ``ingest-run-report``) are always ``raw``: the posture governs only what
is uploaded while the env is built.

Stream contract: stdout is the manifest and NOTHING else (the plugin captures it verbatim) — except
under ``ingest-corpus --extract-only``, which stops before any egress and puts the local extraction
summary there in the manifest's place; every human-facing line — the disclosure preview, per-folder
warnings, errors — goes to stderr.

``ingest-traces`` and ``ingest-corpus`` deliver a grounding part instead of a lease: they print the part
entry (see :mod:`plurai.ingest.parts`) in the manifest's place, and ``ingest-descriptions`` prints the
entries of the agent descriptions it delivered. The commands below describe the lease manifest the other
``ingest-*`` commands print.

Single-archive manifest (the SDK↔backend interface the backend build jobs consume) — ARCHIVE-ONLY::

    {"lease_id": "<uuid>", "checksum": "<sha256 of the archive>", "bytes": <archive size>}

The archive still bundles every unit as a tar member named by the pipeline's unit addressing —
doc-examples ``"<family>/<path>"``, scenario-bundle ``"<scenario>/<path>"``, traces the relative
POSIX path under ``traces/`` (its keyed-hash pseudonym under ``anonymized``), corpus ``"chunks"``/``"relationships"``, run-artifacts
``"<scenario>/<file>"`` plus the run-level ``"run_summary.json"``, run-report each rendered report
file's path under the run dir — and the server untars them by
those names. But the manifest carries NO per-member ``{key, checksum, bytes}`` entries: the
whole-archive ``checksum`` pins every member's bytes (it is computed over the exact archive that is
PUT, and the server re-verifies it on PUT + fetch), so per-member descriptors are redundant. The
archive is byte-deterministic: the same members give the same checksum on every run, so the checksum
is also the content address a backend can be asked about before a byte is sent. Set-level overviews (``_TYPES.md`` /
``BUNDLE.md``) are inline finalize params, not archive members. There is NO client-minted
``set_id``/``job_id`` — the server owns the id; the SINGLE archive lease is the whole transport +
idempotency key.

``ingest-run`` adds one more key to that manifest: a ``provenance`` block naming what the run IS OF
(eval set, scenario set, harness fingerprint, agent build, and this SDK's version), so a later replay
of the uploaded artifacts can tell whether they still describe the current world. Every part of it is
optional and absent parts are omitted, never nulled; it is stamped on the printed manifest only and
never rides a storage request, so it cannot affect whether the upload succeeds.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextlib
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO
from uuid import UUID

import httpx

from plurai import client_handshake, exit_codes, network_refusal
from plurai import __version__, diagnose as diagnose_commands, login as login_commands
from plurai import _private_files
from plurai._platform import is_windows, utf8_stdio
from plurai.egress import PLUGIN_WORKDIR
from plurai.gates.cli import INGEST_OVERRULE_SUBCOMMANDS, OverruleFileError, add_overrule_argument, load_overrules, takes_overrule
from plurai.gates.invocation import RecordTarget, current, held_exit, run_invocation
from plurai.host_effects import host_read_scope, redacted_process_output
from plurai.identity import sdk_identity, sdk_identity_line
from plurai.ingest import backend_commands, fetch_commands, job_commands
from plurai.ingest.anonymize import (
    KEY_DIR_NAME,
    AnonymizationDependencyError,
    Anonymizer,
    DocExampleBinaryError,
    _ensure_key_dir_gitignored,
    load_or_create_key,
)
from plurai.ingest.archive_cap import (
    ArchiveCapConfigError,
    ArchiveOverCapError,
    check_uncompressed_total,
    resolve_max_uncompressed_bytes,
)
from plurai.ingest.disclosure import (
    IngestAuditError,
    MixedPostureError,
    UploadPreview,
    assert_posture_consistent,
    write_audit_entry,
)
from plurai.ingest.folder_names import (
    PSEUDONYMIZED_FOLDER_KINDS,
    is_regression_folder,
    load_name_map,
    outgoing_text,
    pseudonyms_for,
    record_names,
    reveal,
    write_outgoing_overview,
)
from plurai.ingest.manifest_models import IngestManifest, RunProvenance
from plurai.ingest.parts import (
    SCAN_CONTRACT_NAME,
    PartMismatchError,
    PartSetChangedError,
    DESCRIPTION_FILES,
    BuiltPart,
    PartEntry,
    PartKind,
    disclose_parts,
    named_sha256_argument,
    part_lines,
    plan_parts,
    planned_entry,
    render_parts_disclosure,
    require_expected_sha256,
    require_expected_sha256_by_name,
    sha256_argument,
    sha256_of,
    upload_disclosed,
)
from plurai.ingest.claim_scrub import Claim, claim_scrub_unit, load_claims
from plurai.ingest.scrub_map import MEMBER_NAME as SCRUB_MAP_MEMBER
from plurai.ingest.scrub_map import ScrubMapError, ScrubRecorder, load_declared_names, scrub_trace_tree
from plurai.issue_turn import ISSUE_TURN_FILENAME, issue_turn_problem
from plurai.ingest.posture import POSTURE_CONFIG_KEY, RUN_KINDS, AnonymizationPosture, PostureConfigError, PostureResolution, resolve_posture
from plurai.ingest.storage_client import (
    MAX_OBJECT_NAME_LENGTH,
    PartUploadUnsupportedError,
    StorageClient,
    StorageConfigError,
    StorageContentUndelivered,
    StorageCreditStopError,
    StorageUploadError,
)
from plurai.orchestrator._retry import NetworkRefused
from plurai import readers, scoring, verdicts
from plurai.rag_extract import CorpusExtractionDependencyError, CorpusRootsError, CorpusScopeError
from plurai.run_artifacts import (
    CLI_SESSION_DIRNAME,
    PREVIOUS_ATTEMPT_DIRNAME,
    REGRESSION_REPLAYS_FILENAME,
    REGRESSION_SCENARIO_PREFIX,
    REQUIRED_ARTIFACT_FILES,
    RUN_SUMMARY_FILENAME,
    SCENARIO_ARTIFACT_FILES,
)
from plurai.trace_adapter import ADAPTER_DIR_NAME
from plurai.trace_convert import (
    DEFAULT_SOURCE_FORMAT,
    RESUME_PIN_FILENAME,
    ROLES_FILENAME,
    TRACE_FORMAT_REGISTRY,
    TRANSCRIPT_FILENAME,
    ConversionResult,
    PostureRefusedError,
    ThreadConversion,
    TraceConversionError,
    convert_trace_export,
    mid_turn_tail_problem,
    render_trace_export,
)
from plurai.trace_extract import TraceExportError, TraceExtractionDependencyError, trace_selection_keys
from plurai.trials import listed_trial_dirs
from plurai.user_path import expand_user_path


class IngestInputError(Exception):
    """A routine, user-actionable input problem (absent/empty/malformed extraction
    output), surfaced as a clean fail-closed error rather than a raw traceback — e.g.
    a missing ``chunks.json``, an empty trace/bundle directory, or an unreadable
    ``families.json``. Distinct from a genuine bug, which still tracebacks loudly."""


# Fail-closed / user-actionable errors: the CLI presents these as a clean one-line
# error + nonzero exit (not a raw traceback). Every message already carries its own
# actionable install/fix hint. Extractor deps (plurai-extract / sink clients) are
# optional and lazy-imported, so a plain `pip install plurai-simulation-sdk` + ingest surfaces here.
_FAIL_CLOSED_ERRORS = (
    CorpusExtractionDependencyError,
    CorpusRootsError,  # the roots given cannot be one scope (two share a name) — the message names both and the fix
    CorpusScopeError,  # the scan read a suffix ingest has no reader for — the message names it
    TraceExtractionDependencyError,
    TraceExportError,  # a selection LangSmith rejected, an empty export, or an out dir holding files it did not write
    AnonymizationDependencyError,
    DocExampleBinaryError,
    StorageConfigError,
    StorageUploadError,  # the backend refused the upload with a 4xx (checksum, size, parts, auth): a retry cannot fix it
    PartUploadUnsupportedError,  # the backend takes no part uploads, the only way this SDK sends content
    IngestInputError,
    ScrubMapError,  # the scrub map cannot be written faithfully, or the declared names cannot be read
    PartSetChangedError,  # a part changed after it was disclosed, so nothing was uploaded
    PartMismatchError,  # a part built now is not the one the approved plan named — refused before anything is disclosed
    IngestAuditError,  # bytes already egressed but the audit write failed → surface the actionable one-liner
    PostureConfigError,  # posture could not be resolved from plurai.config.yaml — fail closed before egress
    MixedPostureError,  # this run's posture disagrees with a prior upload in this env — fail closed before egress
    TraceConversionError,  # a trace export could not be converted faithfully, or the env's posture refuses it
    ArchiveOverCapError,  # the archive would exceed the server's total-uncompressed cap — refused before egress
    ArchiveCapConfigError,  # the cap override is not a positive integer — refused before egress
)

#: Where an ingest command's gate record lives: ``<workspace-root>/.plurai/gate_authority/<command>.json``.
#: Under the private, self-ignoring ``.plurai/`` beside the disclosure audit, and never beside the
#: command's own output — a bundle directory is published by ``ingest-bundle`` and a run directory
#: holds the runner's own record, and an internal record must not ride either. Written only when the
#: invocation has something to record (an ``--overrule`` or a finding), so a plain invocation creates
#: nothing under the workspace root.
GATE_RECORD_DIRNAME = 'gate_authority'

#: What an over-cap refusal suggests when the command has no narrower selection to offer.
_DEFAULT_OVER_CAP_REMEDY = 'Upload less: narrow the input this command reads, then re-run.'


@dataclass(frozen=True)
class Unit:
    """One upload unit: an on-disk (soon-to-be-anonymized) file + its manifest key.

    A ``verbatim`` unit rides in the archive as written: the scrub never touches it. Only material that is
    already the customer's own description of the agent is verbatim (see :func:`domain_unit`).
    """

    key: str
    path: Path
    verbatim: bool = False


# --------------------------------------------------------------------------- #
# Unit builders (pure: map an extractor's output dir → keyed upload units)      #
# --------------------------------------------------------------------------- #


#: Set-level overview files — supplied to finalize as INLINE params, NOT lease entries.
#: Excluded from the doc/bundle manifests.
DOC_EXAMPLE_SET_OVERVIEW = '_TYPES.md'
BUNDLE_SET_OVERVIEW = 'BUNDLE.md'


def corpus_units(out_dir: str | Path) -> list[Unit]:
    """RAG/SQL corpus → exactly two whole-array objects (the storage wire contract).

    Each unit's stored object is a SINGLE JSON array (the extractor's ``json.dumps([...])``
    output), uploaded as one object → one lease — NOT JSONL, NOT one-object-per-chunk:

    * ``key="chunks"`` — REQUIRED: the JSON array of ``Document.to_dict()`` objects.
    * ``key="relationships"`` — OPTIONAL: the JSON array of relationship objects.

    Fails closed if ``chunks.json`` is absent (an incomplete extraction must not publish
    a corpus with no chunks).
    """
    out = Path(out_dir)
    chunks = out / 'chunks.json'
    if not chunks.exists():
        raise IngestInputError(f'corpus ingest requires the extractor output {chunks} (a JSON array of chunks); none found')
    units = [Unit(key='chunks', path=chunks)]
    relationships = out / 'relationships.json'
    if relationships.exists():
        units.append(Unit(key='relationships', path=relationships))
    return units


#: The most a corpus domain file may hold, refused before anything is uploaded.
MAX_DOMAIN_BYTES = 256 * 1024
#: The member of a corpus part that carries the domain file.
DOMAIN_MEMBER = 'domain.json'
_DOMAIN_KEYS = frozenset({'domain', 'questions'})


def read_domain_file(path: str | Path) -> bytes:
    """The bytes of a corpus ``--domain-file`` once they are known to be a valid domain file.

    The file is a JSON object with an optional ``domain`` object and an optional ``questions`` list of strings, at
    most :data:`MAX_DOMAIN_BYTES`; anything else is refused with the reason.
    """
    source = Path(path)
    try:
        size = source.stat().st_size
        if size > MAX_DOMAIN_BYTES:
            raise IngestInputError(f'--domain-file {source} is {size} bytes; the most a domain file may hold is {MAX_DOMAIN_BYTES}.')
        raw = source.read_bytes()
    except OSError as e:
        raise IngestInputError(f'--domain-file {source} cannot be read: {e}') from e
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise IngestInputError(f'--domain-file {source} must be valid JSON: {e}') from e
    if not isinstance(body, dict):
        raise IngestInputError(f'--domain-file {source} must be a JSON object, got {type(body).__name__}.')
    if unknown := sorted(set(body) - _DOMAIN_KEYS):
        raise IngestInputError(f'--domain-file {source} has unknown keys {unknown}; it takes only "domain" and "questions".')
    if 'domain' in body and not isinstance(body['domain'], dict):
        raise IngestInputError(f'--domain-file {source}: "domain" must be a JSON object.')
    questions = body.get('questions', [])
    if not isinstance(questions, list) or not all(isinstance(question, str) for question in questions):
        raise IngestInputError(f'--domain-file {source}: "questions" must be a list of strings.')
    return raw


def domain_unit(path: str | Path, out_dir: str | Path) -> Unit:
    """The ``domain.json`` unit for a corpus part, from the operator's validated ``--domain-file``.

    The file is derived from the agent descriptions, which travel unscrubbed, so the unit is ``verbatim``: the part
    carries the file's exact bytes and the scrub never meets it. A copy under ``out_dir`` is what the part reads, so
    the operator's file is never rewritten and the bytes disclosed are the bytes sent.
    """
    raw = read_domain_file(path)
    staged = Path(out_dir) / DOMAIN_MEMBER
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(raw)
    return Unit(key=DOMAIN_MEMBER, path=staged, verbatim=True)


def _corpus_json_array(path: Path) -> list:
    """One extractor output file, failing closed on anything that is not a JSON array."""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        raise IngestInputError(f'corpus ingest requires a readable {path}: {e}') from e
    if not isinstance(value, list):
        raise IngestInputError(f'{path} must hold a JSON array of extractor output, got {type(value).__name__}')
    return value


#: Source kinds whose corpus is a CHAIN, not a walkable graph: one record per chunk, linked
#: sequentially, so `relationships ≈ chunks - 1` and `source_documents` counts the ONE file the
#: records came out of. Applying the directory heuristic here calls a healthy record corpus
#: "UNDER-CHUNKED" on every single run (`chunks == source_documents` is false only by accident).
_CHAIN_SOURCE_KINDS = frozenset({'json_records'})

#: SQL and cloud-retriever kinds: the chunking is the SOURCE's, and how many edges a query's rows
#: earn is a property of the data. There is no shape to have an opinion about, so the note reports
#: the numbers and stops — a verdict here would be invented.
_UNVERDICTED_SOURCE_KINDS = frozenset(
    {
        'sqlite',
        'bigquery',
        'snowflake',
        'redshift',
        'azure_sql',
        'databricks',
        'azure_search',
        'vertex_search',
        'bedrock_kb',
        'langchain_vectorstore',
        'llamaindex_vectorstore',
    }
)


def corpus_shape_note(summary: dict[str, int | float | str], source_kind: str | None) -> str:
    """One line about what this corpus's SHAPE means — for the kind it actually came from.

    The UNDER-CHUNKED heuristic (`chunks == source_documents`, or fewer edges than the chunks that owe one) reads
    a DIRECTORY corpus: many files, each split at its headings, cross-linked. On a `json_records`
    corpus every one of those conditions is the healthy shape — one record per chunk, chained, out
    of one file — so applying that heuristic there warns on every healthy record corpus.
    """
    chunks = int(summary.get('chunks') or 0)
    rels = int(summary.get('relationships') or 0)
    docs = int(summary.get('source_documents') or 0)
    # DERIVED when absent or non-numeric, never trusted blindly: a hand-built or older summary
    # without the key would fall through the directory branch to "walkable" on the very numbers
    # that define under-chunked, and would render a literal `None relationship(s) per chunk`.
    per = summary.get('relationships_per_chunk')
    if not isinstance(per, (int, float)) or isinstance(per, bool):
        per = round(rels / chunks, 2) if chunks else None
    numbers = f'{chunks} chunk(s), {rels} relationship(s), {docs} source unit(s)'

    if source_kind in _CHAIN_SOURCE_KINDS:
        # A chain of N nodes has N-1 edges. Only ZERO edges over several chunks is a failure —
        # that means nothing linked at all.
        expected = max(0, chunks - 1)
        if chunks > 1 and rels == 0:
            return (
                f'shape: {chunks} record chunk(s) and NO relationships — a json_records corpus should chain '
                f'(~{expected} edges). Nothing linked; re-check the extractor output before uploading.'
            )
        if rels < expected:
            # A PARTIAL chain. Not the directory verdict — nothing here is under-chunked — but
            # printing "Healthy" over 1 link across 50 records would contradict the expectation
            # stated two lines above it.
            return (
                f'shape: json_records chain is INCOMPLETE — {rels} of ~{expected} expected links across '
                f'{chunks} record chunk(s) from {docs} source file(s). Re-check the extractor output.'
            )
        return f'shape: chain expected for json_records — {chunks} record chunk(s), {rels} link(s) from {docs} source file(s). Healthy.'
    if source_kind in _UNVERDICTED_SOURCE_KINDS:
        return f'shape: {numbers} — no shape verdict for source_kind={source_kind!r}.'
    if source_kind == 'directory':
        if per is None:
            return f'shape: {numbers} — no chunks, so no shape verdict.'
        if chunks and chunks == docs:
            return (
                f'shape: UNDER-CHUNKED — {chunks} chunk(s) from {chunks} source document(s), i.e. one chunk per file, '
                'so the graph has nothing to walk. Re-run with a smaller --chunk-size.'
            )
        if per < 1:
            return (
                f'shape: UNDER-CHUNKED — {per} relationship(s) per chunk that owes one (fewer edges than those chunks), '
                'unwalkable at the planner 2-hop minimum. Re-run with a smaller --chunk-size; a Markdown document with no '
                'headings stays one chain at any size, so give it section headings.'
            )
        return f'shape: walkable — {chunks} chunk(s) from {docs} source document(s), {per} relationship(s) per chunk that owes one.'
    return f'shape: {numbers} — source_kind unknown, no verdict.'


def corpus_extraction_summary(out_dir: str | Path, source_kind: str | None = None) -> dict[str, int | float | str | dict[str, int]]:
    """Local health of a just-extracted corpus — what ``--extract-only`` prints in the manifest's place.

    Counts what is ON DISK (the same two files :func:`corpus_units` would upload), so the numbers
    describe exactly the corpus that would egress. Chunks of one source file share their ``name`` and
    differ by ``chunk_idx``, so DISTINCT names are the source-document count — which makes both shapes
    the corpus health gate rejects visible before a byte leaves: ``chunks == source_documents`` (one
    chunk per file, so the graph has nothing to walk) and ``relationships_per_chunk < 1`` (fewer edges
    than the chunks that owe one). ``relationships_per_chunk`` divides by the chunks that OWE an edge: for
    a ``directory`` source a file read in token windows is a chain (N chunks, N-1 edges), so its first
    chunk owes none; every other chunk, and every chunk of any other source kind, owes one. The verdict
    stays with the gate; this reports the numbers it keys on.

    Fails closed (:class:`IngestInputError`) when the extraction produced no usable chunk, so
    "nothing usable" exits 65 like every other unusable extractor output instead of reporting a
    healthy-looking zero.
    """
    out = Path(out_dir)
    chunks = _corpus_json_array(out / 'chunks.json')
    relationships_path = out / 'relationships.json'
    relationships = _corpus_json_array(relationships_path) if relationships_path.exists() else []
    if not chunks:
        raise IngestInputError(f'corpus extraction produced 0 chunks under {out}; check the source path / glob / query — an empty corpus cannot be built')
    # Only a REAL string name folds into `per_document` — coercing a missing/None name via
    # `str(...)` would spell it `'None'`, silently merging that chunk's count into any document
    # actually named "None" (two unrelated things collapsing into one bucket). A chunk with no
    # name at all counts separately, under a key no source document's own name can ever equal.
    per_document: dict[str, int] = {}
    unnamed_chunks = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        name = chunk.get('name')
        if isinstance(name, str):
            per_document[name] = per_document.get(name, 0) + 1
        else:
            unnamed_chunks += 1
    source_documents = {chunk.get('name') for chunk in chunks if isinstance(chunk, dict)}
    # The edges a directory corpus owes: a file read in token windows is a chain (N chunks, N-1 edges), so
    # its first chunk owes none. A markdown corpus is unaffected; without this, a few small non-markdown
    # files beside walkable guides read as under-chunked, and no --chunk-size clears a chain.
    owing = len(chunks)
    if source_kind == 'directory':
        from plurai_extract.sources.directory import SECTION_CHUNKED_SUFFIXES

        owing -= sum(1 for name in source_documents if PurePosixPath(str(name)).suffix.lower() not in SECTION_CHUNKED_SUFFIXES)
    summary: dict[str, int | float | str | dict[str, int]] = {
        'chunks': len(chunks),
        'relationships': len(relationships),
        'source_documents': len(source_documents),
        'relationships_per_chunk': round(len(relationships) / max(1, owing), 2),
        # Additive: each source document's chunk count, so the operator sees WHICH documents the total is made of.
        'per_document': dict(sorted(per_document.items())),
    }
    # Additive, and only when NONZERO: a chunk with no name is real (a source that doesn't set
    # one) but must never be attributed to any document, `"None"`-named or otherwise.
    if unnamed_chunks:
        summary['unnamed_chunks'] = unnamed_chunks
    # Additive, and only when KNOWN: the four numbers above are the stable contract (an older
    # plugin reads exactly them, so `relationships_per_chunk` keeps its key while meaning edges per
    # chunk that owes one), and a `source_kind` invented here would be worse than absent.
    if source_kind:
        summary['source_kind'] = source_kind
    return summary


def trace_units(out_dir: str | Path) -> list[Unit]:
    """Trace export → one unit per file under ``traces/`` (key = its RELATIVE POSIX path).

    Nested layouts are supported (the key is the path under ``traces/``, not a flat
    ``<trace_id>.json`` — flat is just the common case). One lease per file.
    """
    traces_dir = Path(out_dir) / 'traces'
    return [Unit(key=p.relative_to(traces_dir).as_posix(), path=p) for p in sorted(traces_dir.rglob('*.json')) if p.is_file()]


def pseudonymize_trace_units(scrubber: Anonymizer, units: list[Unit]) -> list[Unit]:
    """Trace units renamed, on disk and as archive members, to their keyed-hash pseudonyms.

    A trace file is named after the id it holds, and a LangSmith conversation export names it
    after the thread id, which is often a phone number, an email or a channel handle. The scrub
    rewrites file content, never the name, so under ``anonymized`` each file becomes
    ``<pseudonym>.json`` (flat under ``traces/``), the pseudonym being the one the scrub writes
    for the unit's key. The local file is renamed too, because the backend's brief names trace
    files by their member names and the caller resolves those names under ``traces/``. The
    export's local ``manifest.json`` has its ``selection.thread_files`` follow the rename, so each
    thread id still maps to its file; the audit record's ``member_sources`` maps every new name
    back to the old one.
    """
    if not units:
        return units
    planned = [(unit, f'{scrubber.pseudonym(unit.key.removesuffix(".json"))}.json') for unit in units]
    names = [name for _, name in planned]
    if len(set(names)) != len(names):
        raise IngestInputError('two trace files map to one pseudonymous name; rename one of them and re-run')
    traces_dir = units[0].path
    for _ in PurePosixPath(units[0].key).parts:
        traces_dir = traces_dir.parent
    taken = [name for unit, name in planned if (traces_dir / name) != unit.path and (traces_dir / name).exists()]
    if taken:
        raise IngestInputError(f'{traces_dir} already holds {taken[:3]}, the pseudonymous name(s) of other trace files; clear the out dir and re-run')
    renamed: list[Unit] = []
    for unit, name in planned:
        target = traces_dir / name
        if target != unit.path:
            unit.path.rename(target)
        renamed.append(Unit(key=name, path=target))
    _record_trace_pseudonyms(traces_dir.parent / 'manifest.json', {unit.key: name for unit, name in planned})
    return renamed


def _record_trace_pseudonyms(manifest_path: Path, renamed: dict[str, str]) -> None:
    """Point the export's local manifest at the renamed files. A manifest with no ``thread_files`` is left as it is."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    selection = manifest.get('selection') if isinstance(manifest, dict) else None
    if not isinstance(selection, dict) or not isinstance(selection.get('thread_files'), dict):
        return
    selection['thread_files'] = {thread: renamed.get(name, name) for thread, name in selection['thread_files'].items()}
    tmp = manifest_path.with_name(f'{manifest_path.name}.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(manifest_path)


def doc_example_units(out_dir: str | Path) -> list[Unit]:
    """Doc examples → one unit per ``families.json`` entry (key ``<family>/<path>``).

    Includes each family's ``FAMILY.md`` (it is an ordinary families.json entry) but
    EXCLUDES the set-level ``_TYPES.md`` overview (an inline finalize param, not a lease
    — and the extractor already keeps it out of families.json as a loose file; the skip
    is a belt-and-suspenders guard).

    Each unit file holds the example's DECODED bytes, not its ``families.json`` entry: the
    backend writes every archive member verbatim into the set, and document generation copies
    those files as its few-shot examples, so a JSON wrapper here would reach the generator as
    the example itself (a PNG would arrive as base64 text). ``families.json`` is only the
    extractor's local hand-off. Scrubbing still reaches inside the content
    (:func:`_doc_example_scrub_unit`): a supported binary asset (image/PDF) is best-effort
    scrubbed (OCR-redact + rasterize), failing closed only on an unsupported/corrupt asset or
    a missing binary dependency.
    """
    out = Path(out_dir)
    families = out / 'families.json'
    try:
        entries = json.loads(families.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as e:
        raise IngestInputError(f'doc-examples ingest requires a readable {families}: {e}') from e
    unit_dir = out / '_units'
    unit_dir.mkdir(exist_ok=True)
    units = []
    for i, entry in enumerate(entries):
        if Path(entry['path']).name == DOC_EXAMPLE_SET_OVERVIEW:
            continue  # set-level overview → inline finalize param, never a lease
        key = f'{entry["family"]}/{entry["path"]}'
        try:
            content = base64.b64decode(entry['content_b64'], validate=True)
        except (binascii.Error, ValueError) as e:
            raise IngestInputError(f'{families}: content_b64 of {key!r} is not valid base64: {e}') from e
        # The unit file keeps the example's suffix: the binary scrub names it when it refuses the asset.
        path = unit_dir / f'doc_{i:04d}{Path(entry["path"]).suffix}'
        path.write_bytes(content)
        units.append(Unit(key=key, path=path))
    return units


#: The saved-run artifact files ``score_run`` consumes (``verdicts.json`` among them: the verdicts the backend's
#: readers take); anything else in a scenario folder (sidecars other tools wrote) is local working state and
#: never uploads. THE list the runner writes (``plurai.run_artifacts``), not a second copy of
#: it — a private restatement here is how "what one attempt produces" drifts between the writer
#: and the uploader.
RUN_ARTIFACT_FILES = SCENARIO_ARTIFACT_FILES

#: Names that identify a directory as the RUNNER's own per-scenario output — the uploadable
#: artifacts, plus the crash record written instead of them. Evidence for the "lost scenario vs.
#: somebody else's directory" test ONLY; the upload list stays ``RUN_ARTIFACT_FILES``.
_SCENARIO_EVIDENCE_FILES = (*RUN_ARTIFACT_FILES, 'error.json')


#: The one nested folder a scenario directory may carry that is itself a scenario: the stability
#: retry, a whole second attempt with its own transcript and its own verdicts. The name is shared
#: with the backend's upload store, which admits exactly this one nested item name.
RETRY_FOLDER = 'stability_retry'


def _retry_units(folder: Path, names: Sequence[str]) -> tuple[list[Unit], str | None]:
    """``(units, why it was skipped)`` for a scenario folder's ``stability_retry/``.

    ``run_units`` walks only the run directory's TOP level, and a retry lives one level down inside
    its scenario's folder, so the retry is packed here instead.

    Keyed ``<scenario>/stability_retry/<file>``, which is the scenario's own key plus the segment —
    the only nested shape the backend's item jail admits.

    Same artifact rules as the primary attempt, deliberately: a retry with no transcript is not a
    retry, and an unreadable artifact is a loss rather than something to upload and fail on mid-
    egress. It is reported through the CALLER's skip channel rather than raising, because a retry
    that cannot be packed must not cost the primary attempt its upload.
    """
    if RETRY_FOLDER not in names:
        return [], None
    retry = folder / RETRY_FOLDER
    if not _safe_is_dir(retry):
        return [], f'{folder.name}/{RETRY_FOLDER} (named but not a directory)'
    retry_names = _entry_names(retry)
    if retry_names is None:
        return [], f'{folder.name}/{RETRY_FOLDER} (unreadable — cannot list its contents)'
    present = [name for name in RUN_ARTIFACT_FILES if name in retry_names]
    if 'transcript.json' not in present:
        return [], f'{folder.name}/{RETRY_FOLDER} (no transcript.json)'
    unreadable = [name for name in present if not _safe_is_file(retry / name)]
    if unreadable:
        return [], f'{folder.name}/{RETRY_FOLDER} (unreadable artifact: {", ".join(unreadable)})'
    return [Unit(key=f'{folder.name}/{RETRY_FOLDER}/{name}', path=retry / name) for name in present], None


def _entry_names(folder: Path) -> list[str] | None:
    """Names directly inside *folder*, or ``None`` when it cannot be listed (EACCES, …).

    Listing ONCE and matching names replaces a ``(folder / name).exists()`` probe per
    artifact, which is wrong twice: ``Path.exists()`` follows symlinks, so a DANGLING
    artifact symlink reads as absent and quietly demotes the folder to "not a scenario"; and
    it re-raises EACCES, so a mode-000 folder blows ``run_units`` up with a bare
    ``PermissionError``. Listing puts both in the LOUD bucket instead.
    """
    try:
        return sorted(p.name for p in folder.iterdir())
    except OSError:
        return None


def _safe_is_dir(path: Path) -> bool:
    """``path.is_dir()`` that answers False instead of raising on an unreadable parent."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    """``path.is_file()``; False for a DANGLING symlink and for anything we may not stat."""
    try:
        return path.is_file()
    except OSError:
        return False


def _required_input_dir(value: str | Path, flag: str, *, walks: str, hint: str) -> Path:
    """The directory a directory-addressed verb was aimed at — or a refusal naming the flag.

    ONE reader for every directory-addressed arm. Both refusals follow from two facts about
    ``Path``:

    * EMPTY is refused BY NAME, before ``Path`` gets it. ``Path('')`` is ``Path('.')``, so an
      empty value passes the ``is_dir`` test below AS THE CURRENT DIRECTORY and gets walked —
      and the live source of an empty value is a shell reader (``--run-dir "$(cat
      "$WORKDIR/.last_run")"`` expands to nothing when no runner wrote the pointer).
      "Nothing was named" and "the current directory was named" are different requests and only
      the second one may be scanned.
    * A path that is not a directory is refused rather than walked as an empty one: ``rglob`` on
      a missing directory yields nothing, so a mistyped path would publish ZERO units and report
      success.

    ``walks`` says what the caller would have done with the wrong directory, because the cost is
    what makes this a refusal rather than a default; ``hint`` carries the one sentence that is
    about THAT flag's own upstream, which a shared message must not lose.
    """
    if not str(value).strip():
        raise IngestInputError(f'{flag} was EMPTY. An empty path is the current directory, so this would have {walks} under {Path.cwd()}. {hint}')
    root = Path(value)
    if not root.is_dir():
        raise IngestInputError(f'{flag} must name an existing directory; {root} is not one')
    return root


def _has_scenario_evidence(names: list[str]) -> bool:
    """True when any entry name CONTAINS a scenario-evidence filename.

    Substring, not equality, on purpose: an in-flight ``rsync`` stages
    ``.transcript.json.k3Jd9x`` before the final rename, and a half-finished copy must not
    read as "never was a scenario folder".
    """
    return any(marker in name for name in names for marker in _SCENARIO_EVIDENCE_FILES)


def _lost_scenario_reason(folder: Path, names: list[str]) -> str | None:
    """Why an artifact-free *folder* is still a LOST scenario — or ``None`` if it genuinely
    is not the runner's output and may be ignored quietly.

    The quiet bucket exists for exactly one shape: a directory some OTHER tool wrote into
    ``$OUT`` — the plugin's report renderer drops ``charts/*.svg`` there, and counting that
    as a failed scenario would make every rendered run look like it had lost one. These three
    artifact-free shapes are not that, and every one of them is a real loss:

    * **empty** — ``run_simulation._run_one`` clears the artifact names BEFORE running a
      scenario (re-run hygiene), so a re-run killed mid-flight (Ctrl-C, OOM, a heavy
      ``--concurrency``) leaves its pre-existing scenario dirs with nothing at the top level.
      A copy that created the directories but not yet the files lands on the same shape.
    * **an interrupted re-run whose PREVIOUS attempt survived** — the same shape, except the
      superseded attempt is sitting in ``previous_attempt/`` (the runner displaces rather than
      deletes). Named separately because the remedy is different: the measurement still
      exists, and re-running is a choice rather than the only option.
    * **artifact names still in a partial/temp form** — caught by the substring test above.
    * **a nested run tree** — ``$OUT/run_a/scenario_09/transcript.json``. Ignoring
      the parent drops a whole run in one quiet line, and because a flat sibling can still
      carry the upload, the fail-closed guard never fires: the loss resurfaces much later as
      ``score_run``'s opaque "has no scenario folder".
    """
    if not names:
        return 'empty — interrupted mid-run, or a copy that never finished'
    if _has_scenario_evidence(names):
        return 'artifact names present only in partial/temp form — a copy still in flight?'
    preserved_names = _entry_names(folder / PREVIOUS_ATTEMPT_DIRNAME) if PREVIOUS_ATTEMPT_DIRNAME in names else None
    if preserved_names is not None and _has_scenario_evidence(preserved_names):
        # Checked BEFORE the nested-run-tree walk below, which would otherwise describe this
        # exact shape as "the scenario folders are one level down" and send the operator to
        # re-point --out at a directory that is not a run.
        return (
            f'this attempt produced nothing — the attempt it superseded is preserved under '
            f'{PREVIOUS_ATTEMPT_DIRNAME}/ (a re-run interrupted, or one that failed before writing). '
            f'Re-run the scenario, or upload the preserved attempt deliberately by moving it up'
        )
    for name in names:
        child = folder / name
        if not _safe_is_dir(child):
            continue
        child_names = _entry_names(child)
        if child_names is None:
            # UNREADABLE, not empty. ``_entry_names`` answers None for an EACCES/OSError listing;
            # treating that as "nothing there" would report a scenario folder as somebody else's
            # directory and drop it from the archive on a quiet line.
            return f"unreadable subdirectory {name}/ — cannot tell whether this scenario's artifacts are inside it"
        if _has_scenario_evidence(child_names):
            return f'nested run tree — the scenario folders are one level down (e.g. {name}/); point --out at that directory'
    return None


def _run_summary_unit(root: Path) -> list[Unit]:
    """The run-LEVEL ``run_summary.json`` as a one-element unit list — empty when the run has none.

    This is the run's own verdict about itself: counts, ``scored``, ``ended_early_scenarios``,
    ``measurement_clean`` / ``degraded_scenarios`` / ``measurement_integrity``. Without it the
    backend can read every scenario folder and still not know how many scenarios the RUN thought
    it had, nor that a row it is about to score was never measured — so a suite report reconciles
    its own count against nothing and reads a degraded row as an ordinary agent failure.

    OPTIONAL, in both directions. A run legitimately lacks one — the runner writes it last, so a
    run killed mid-flight has none — and the backend's reader answers ``None`` rather than
    raising. Absence is REPORTED on stderr rather than passed over: "this uploader wrote none" and
    "this run was clean" are the two answers the whole integrity axis turns on, and a silent skip
    is how they get confused.

    WHERE IT LANDS. The server places every archive member under the set's ``scenarios/``
    container; a member cannot name the set root, because a ``../`` member name is refused as a
    traversal. The backend reads the member from that container path. A loose file beside the scenario
    directories is neither a scenario nor an error.

    READABILITY IS PART OF PRESENCE HERE, because the reader on the other side does not degrade.
    ``_safe_is_file`` alone would ship a 0-byte or truncated file — a copied/rsynced ``$OUT``, an
    interrupted write, a hand-edit — and ``read_run_summary`` RAISES on one
    (``'<file> is present but unreadable'``; likewise for JSON that is not an object) rather than
    answering ``None``. So a file the backend would have shrugged off as absent becomes a hard
    error at score time. Parsing it here costs a few KB of local read and keeps the asymmetry out:
    this module already goes to lengths to keep a half-copied SCENARIO folder out of the archive
    (:func:`_lost_scenario_reason`), and fail-closed-before-egress means the same thing for the
    run-level member.
    """
    summary = root / RUN_SUMMARY_FILENAME
    if not _safe_is_file(summary):
        print(
            f'plurai ingest-run: no readable {RUN_SUMMARY_FILENAME} in {root} — uploading the scenario folders '
            "alone; the backend reports the run's own verdict as unavailable rather than reading its absence "
            'as a clean run',
            file=sys.stderr,
        )
        return []
    try:
        parsed = json.loads(summary.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        # ValueError covers both halves of "not JSON": JSONDecodeError (0-byte, truncated,
        # hand-edited) and UnicodeDecodeError, which is a ValueError and NOT an OSError.
        detail = f'{type(exc).__name__}: {exc}'
    else:
        detail = None if isinstance(parsed, dict) else f'top-level JSON {type(parsed).__name__}, not an object'
    if detail is not None:
        # Same channel and same shape as the absence line above, because the OUTCOME is the same
        # one: the run's own verdict does not reach the backend. What differs is that skipping
        # here is a CHOICE — shipping it would hand the reader a file it raises on, turning a
        # benign gap into a failed score.
        print(
            f'plurai ingest-run: {RUN_SUMMARY_FILENAME} in {root} is present but unreadable ({detail}) — skipping it '
            "and uploading the scenario folders alone; the backend's reader raises on an unparseable summary, so "
            "shipping it would fail the score outright instead of reporting the run's verdict as unavailable",
            file=sys.stderr,
        )
        return []
    return [Unit(key=RUN_SUMMARY_FILENAME, path=summary)]


def _listed_trial_dirs(root: Path) -> frozenset[str]:
    """The trial run directories ``root``'s own ``run_summary.json`` lists (``trials.dirs``).

    A ``--trials N`` run keeps trials 2..N as complete runs at ``$OUT/trial_<k>/``; ``$OUT`` itself
    is trial 1, and each listed trial uploads as ``<scenario>/trial_<k>`` (:func:`_trial_units`). Only
    the directories the summary NAMES are this run's trials — one it does not name (an older run's
    leftover) stays on the loud path below.
    """
    try:
        parsed = json.loads((root / RUN_SUMMARY_FILENAME).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return frozenset()
    return listed_trial_dirs(parsed)


def run_units(run_dir: str | Path) -> list[Unit]:
    """Finished run → one unit per saved artifact file (key ``<scenario>/<file>``), plus the
    run-level ``run_summary.json`` when the run wrote one (:func:`_run_summary_unit`) and each
    listed trial's folders (key ``<scenario>/trial_<k>/<file>``, :func:`_trial_units`).

    ``run_dir`` is the runner's ``$OUT``: one folder per scenario. The archive mirrors the two
    shapes the server import admits, and nothing else:

    * **completed** — ``transcript.json`` plus at least one MATERIAL artifact
      (``final_world_state.json`` / ``tool_events.json``). Uploaded as those files.
    * **failed** — an ``error.json``, for a scenario that errored during the run. Uploaded as
      the error record plus any material the run got as far as writing, and NEVER its
      transcript: the server reads a folder's status off which description file it carries, so
      including a transcript would file the row as ``completed`` and send ``score_run`` at a
      run that never finished.

    A folder holding NEITHER is refused by name — that is what keeps an empty or half-copied
    directory out of the archive.

    A failed folder is admitted rather than dropped at the client: the server accepts one and
    records ``status: 'failed'`` on its manifest row, so the scenarios most worth diagnosing — an
    errored run is the one case nothing else can explain — reach the backend. They still cannot
    be SCORED: ``score_run`` refuses such a row by name, and ``suite_diagnostics`` reports it
    rather than analysing it.

    An ``error.json`` WINS over a transcript in the same folder. A run that died after writing
    some artifacts is still a run that died, and its transcript is not a completed
    conversation — so the folder uploads as failed and that transcript does not reach the
    backend. Stated rather than silently handled: it is a real loss of evidence, and it is the
    lesser one. Filing the row ``completed`` would send ``score_run`` at a run that errored,
    which is a wrong VERDICT rather than a missing artifact.

    This is the client half of score_run's by-reference path: upload the run ONCE, then
    ``score_run(run_upload_id=...)`` per scenario instead of shipping the artifact dicts
    on every call.

    Two DIFFERENT exclusions, reported differently. Anything that could be a LOST scenario is
    **skipped** with a warning on stderr — an archive that silently under-represents the run
    would misdirect score_run's later "has no scenario folder" error at the upload instead of at
    the scenario's own failure. That bucket is deliberately wide: an errored or incomplete
    folder, an EMPTY one (the runner clears the artifact names before each scenario, so an
    interrupted re-run leaves exactly that — with the attempt it superseded preserved beside it
    in ``previous_attempt/``), a half-copied one, a folder whose scenarios sit one level DOWN (a
    nested run tree), and one we cannot read.

    A ``previous_attempt/`` folder is never uploaded. It is a SUPERSEDED attempt: filing it as
    this run's result would report a measurement the run did not take.

    Only a directory that positively is somebody ELSE's is **ignored** on a quiet line: it holds
    files but no scenario evidence at any of those shapes — ``$OUT`` is where the plugin's report
    renderer writes ``charts/``, and counting that as a failed scenario would make every rendered
    run look like it had lost one. The test is structural, not a name denylist, so the next
    directory the renderer adds is ignored too — while the three false-negative shapes above stay
    loud (:func:`_lost_scenario_reason`).

    Of the run DIRECTORY itself only two shapes are refused outright: an EMPTY ``run_dir`` (which
    is the current directory in ``Path`` terms, not a request) and one that is not a directory.
    Past that, it fails closed only when NOTHING is uploadable — no completed folder AND no
    failed one. An all-failed archive is deliberately NOT a failure: it scores nothing, but it is
    exactly the run a diagnosis is wanted for, and refusing it would put the client back in front
    of the server contract it is supposed to mirror.
    """
    # Both refusals belong to the CLASS, not to this flag: :func:`_required_input_dir` owns the
    # empty-is-the-cwd rule and the is-a-directory rule for every directory-addressed verb here,
    # so no member of the class is left open.
    root = _required_input_dir(
        run_dir,
        '--run-dir',
        walks='archived whatever scenario-shaped folders happen to sit',
        hint=(
            'Pass the run output dir explicitly; if the value came from a $(cat ...) of the run pointer, that pointer is '
            'missing — the runner writes it only for a run that ran rows.'
        ),
    )
    units: list[Unit] = []
    skipped: list[str] = []
    ignored: list[str] = []
    failed: list[str] = []
    # Trials 2..N of this run: complete runs of their own, not scenario folders here — packed below.
    trial_dirs = _listed_trial_dirs(root)
    for folder in sorted(p for p in root.iterdir() if _safe_is_dir(p) and p.name not in trial_dirs):
        names = _entry_names(folder)
        if names is None:
            skipped.append(f'{folder.name} (unreadable — cannot list its contents)')
            continue
        present = [name for name in RUN_ARTIFACT_FILES if name in names]
        errored = 'error.json' in names
        if not present and not errored:
            reason = _lost_scenario_reason(folder, names)
            if reason is None:
                ignored.append(folder.name)
            else:
                skipped.append(f'{folder.name} ({reason})')
            continue
        # Named in the listing but not a readable regular file: a dangling symlink, or an
        # entry we may not stat. Uploading it would fail mid-egress, so treat it as a loss.
        # ``error.json`` is checked alongside the artifact names because it is uploadable too —
        # a dangling error record would break the egress exactly like a dangling transcript.
        candidates = [*present, 'error.json'] if errored else present
        unreadable = [name for name in candidates if not _safe_is_file(folder / name)]
        if unreadable:
            skipped.append(f'{folder.name} (unreadable artifact: {", ".join(unreadable)})')
            continue
        material = [name for name in present if name != 'transcript.json']
        # ERRORED WINS, even over a folder that also holds a transcript and a material
        # artifact. An error.json means this scenario did not finish, and a transcript written
        # before it died is not a completed run: the server files a folder's status by which
        # description file it carries, so uploading that transcript would mark the row
        # 'completed' and point score_run at a run that errored.
        if errored:
            upload = ['error.json', *material]
            failed.append(folder.name)
        elif 'transcript.json' in present and material:
            # ``REQUIRED_ARTIFACT_FILES``, not ``RUN_ARTIFACT_FILES``: the upload list is the
            # DISPLACEMENT contract and names every file an attempt COULD leave behind, including
            # the shape-conditional ``pipeline_io.json``. Read as an inventory it would report a
            # missing artifact on every completed DIALOG folder that has ever existed — which is
            # not a fact about that run, and a channel that reports absences which are normal is a
            # channel an operator learns to stop reading. What a hop does not write is still
            # UPLOADED (``present`` above is the superset) and still displaced; only the warning
            # narrows, to the names every shape owes.
            missing = [name for name in REQUIRED_ARTIFACT_FILES if name not in present]
            if missing:
                print(f'plurai ingest-run: {folder.name}: uploading without {", ".join(missing)}', file=sys.stderr)
            upload = present
        else:
            skipped.append(folder.name)
            continue
        units.extend(Unit(key=f'{folder.name}/{name}', path=folder / name) for name in upload)
        # AND THE STABILITY RETRY, if this scenario has one. Uploaded as its own item so the
        # post-repair rescore can reach it; otherwise the retry keeps the pre-repair eval revision
        # for ever and no 'stable on retry' claim holds for the shipped one.
        retry_units, retry_skip = _retry_units(folder, names)
        units.extend(retry_units)
        if retry_skip:
            skipped.append(retry_skip)
    if ignored:
        # Named, so a genuinely misplaced scenario folder is still discoverable — but without the
        # "failed" framing: these hold files, none of them scenario-shaped at any depth we check
        # (e.g. the report renderer's charts/).
        print(f'plurai ingest-run: ignored {len(ignored)} non-scenario folder(s) (no run artifacts): {", ".join(ignored)}', file=sys.stderr)
    if failed:
        # INCLUDED, not skipped — but still named, so the shape of the run stays visible on the
        # same channel the losses are reported on. An operator reading only a scenario count
        # would otherwise have no way to see that a third of the archive cannot be scored.
        print(
            f'plurai ingest-run: including {len(failed)} FAILED scenario folder(s) as error records '
            f'(uploaded for diagnosis; score_run cannot score them): {", ".join(failed)}',
            file=sys.stderr,
        )
    if skipped:
        print(
            f'plurai ingest-run: skipped {len(skipped)} folder(s) with no usable artifact set '
            f'(interrupted, partially copied, unreadable, or a nested run tree): {", ".join(skipped)}',
            file=sys.stderr,
        )
    if not units:
        raise IngestInputError(
            f'run ingest found no uploadable scenario folder under {root} — a folder must hold either '
            'transcript.json plus final_world_state.json/tool_events.json (a completed scenario) or '
            "error.json (one that errored during the run); check this is the runner's output dir ($OUT)"
        )
    # AFTER the fail-closed guard, deliberately: the run-level summary is not a scenario, and
    # counting it would let a $OUT holding nothing BUT a summary pass the guard and egress an
    # archive the server's finalize then rejects for having no scenarios — after the bytes left.
    units.extend(_run_summary_unit(root))
    for trial_dir in sorted(trial_dirs):
        units.extend(_trial_units(root, trial_dir))
    return units


def _trial_units(root: Path, trial_dir: str) -> list[Unit]:
    """A listed trial's scenario folders, packed by :func:`run_units`' own folder rules as the item
    ``<scenario>/<trial_dir>``. Its ``run_summary.json`` stays home (``$OUT``'s ``trials`` block is the
    one source); its report lines name the trial; and a trial with nothing uploadable is one of those
    lines, not a refusal — trial 1 already passed the guard."""
    report = io.StringIO()
    packed: list[Unit] = []
    try:
        with contextlib.redirect_stderr(report):
            packed = [unit for unit in run_units(root / trial_dir) if unit.key != RUN_SUMMARY_FILENAME]
    except IngestInputError as exc:
        print(f'plurai ingest-run: skipped the whole trial ({exc})', file=report)
    for line in report.getvalue().splitlines():
        print(line.replace('plurai ingest-run: ', f'plurai ingest-run: {trial_dir}: ', 1), file=sys.stderr)
    return [Unit(key=unit.key.replace('/', f'/{trial_dir}/', 1), path=unit.path) for unit in packed]


#: The index the plugin's report renderer writes beside the report it rendered: ``{"files": [<relative path>, …]}``.
REPORT_INDEX_FILENAME = 'report_files.json'
#: What the plugin's aggregator writes in place of a report when the run's frame no longer stands.
RUN_STALE_FILENAME = 'run_stale.json'

#: Why a run has no report to attach — the closed vocabulary ``attach_run_report(absent_reason=…)`` takes.
ABSENT_STALE_RUN = 'stale_run'
ABSENT_NOT_RENDERED = 'not_rendered'


def _indexed_report_files(folder: Path) -> list[str] | None:
    """The report files *folder*'s renderer index lists that are still there, relative — ``None`` when there is no index.

    An index that cannot be read, or that names a path outside *folder*, is refused. A listed file that is
    gone is left out and named on stderr: the STALE-banner path deletes ``eval_signal.md`` after the render.
    """
    index = folder / REPORT_INDEX_FILENAME
    if not _safe_is_file(index):
        return None
    try:
        listed = json.loads(index.read_text(encoding='utf-8')).get('files')
    except (OSError, ValueError, AttributeError) as exc:
        raise IngestInputError(f'{index} is unreadable ({exc}); re-render the report') from exc
    if not isinstance(listed, list) or not all(isinstance(name, str) and name for name in listed):
        raise IngestInputError(f'{index} must hold {{"files": [<relative path>, …]}}; re-render the report')
    root = folder.resolve()
    present = []
    for name in sorted(set(listed)):
        path = folder / name
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or not path.resolve().is_relative_to(root):
            raise IngestInputError(f'{index} names {name!r}, which is not inside {folder}')
        if _safe_is_file(path):
            present.append(name)
        else:
            print(f'plurai ingest-run-report: {index} lists {name}, which is no longer there; attaching without it', file=sys.stderr)
    return present


def run_report_units(run_dir: str | Path) -> tuple[list[Unit], str | None]:
    """``(units, absent_reason)`` for a run's rendered report — exactly the files the renderer indexed.

    Keys are the files' paths under ``run_dir`` (``analytics.md``, ``charts/run_summary.svg``,
    ``trial_2/analytics.md``), so the report's own relative links still resolve in the upload. Each
    trial ``run_summary.json`` lists contributes its own index; a trial without one is named on stderr.

    No units means no report, and the reason is typed: :data:`ABSENT_STALE_RUN` when the aggregator
    refused the run (``run_stale.json`` — it wins over any leftover index), else
    :data:`ABSENT_NOT_RENDERED` (a ``recording`` run, or a render that never ran).
    """
    root = _required_input_dir(
        run_dir,
        '--run-dir',
        walks='looked for a rendered report',
        hint='Pass the run output dir explicitly; the runner writes its path to $WORKDIR/.last_run.',
    )
    if (root / RUN_STALE_FILENAME).exists():
        return [], ABSENT_STALE_RUN
    listed = _indexed_report_files(root)
    if not listed:
        return [], ABSENT_NOT_RENDERED
    units = [Unit(key=name, path=root / name) for name in listed]
    for trial_dir in sorted(_listed_trial_dirs(root)):
        trial = root / trial_dir
        trial_listed = None if (trial / RUN_STALE_FILENAME).exists() else _indexed_report_files(trial)
        if not trial_listed:
            print(f'plurai ingest-run-report: {trial_dir}: no rendered report; attaching the others', file=sys.stderr)
            continue
        units.extend(Unit(key=f'{trial_dir}/{name}', path=trial / name) for name in trial_listed)
    return units, None


def _check_pin_is_seedable(pin_path: Path, root: Path) -> None:
    """Refuse a staged resume pin the backend will reject as captured mid-turn.

    The converter applies this same rule when it WRITES a pin, but a bundle arriving here need not
    have come from it: a pin can be edited between conversion and upload, hand-authored, or written
    by another tool. This verb is the last step before the bytes leave the machine and the first
    one that costs money, so the rule is read once more against what is actually on disk. It is the
    shared reader, not a second spelling of the rule — a check the operator can pass locally but
    the backend then fails is worth less than no check at all.
    """
    where = pin_path.relative_to(root).as_posix()
    try:
        payload = json.loads(pin_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise IngestInputError(
            f'{where}: cannot be read as a resume pin ({exc}); a bundle folder carrying this name must hold the pin JSON the converter writes'
        ) from exc
    messages = payload.get('messages') if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages:
        raise IngestInputError(
            f'{where}: holds no non-empty "messages" list, so there is no conversation for the scenario to resume from; '
            're-convert the trace with `plurai convert-trace`'
        )
    if problem := mid_turn_tail_problem(messages):
        raise IngestInputError(
            f'{where}: {problem} The backend applies this rule during generation — after this upload, once a job is '
            'running — so the bundle is refused here instead. Re-convert the trace with `plurai convert-trace` (which '
            'offers --trim-mid-turn-tail for a trace captured mid-turn), or fix the pin.'
        )


def bundle_units(bundle_dir: str | Path) -> list[Unit]:
    """Scenario bundle → one unit per record file (key = RELATIVE POSIX ``<scenario>/<path>``).

    Includes each scenario's ``ROLES.md`` but EXCLUDES the set-level ``BUNDLE.md``
    overview (an inline finalize param, not a lease), and a regression folder's
    ``replays.json`` (its record-time rate, written beside the records). There is no SDK
    bundle extractor; the bundle is a pre-materialized directory whose relative paths are
    the addressing.

    Every resume pin in the tree is opened on the way past, and one the backend cannot seed from
    stops the upload before any of it is egressed. A regression folder's ``issue_turn.json`` — the
    recorded bad reply the backend replays against the check that owns the regression — is carried
    the same way, and one it cannot replay stops the upload too.

    The two directory refusals are :func:`_required_input_dir`'s, shared with ``run_units``.
    This verb is the SHARPER member of that class: it has no scenario-shaped-folder test at all,
    so an empty ``--bundle-dir`` would walk the whole cwd tree and upload every regular file under
    it as a bundle record.
    """
    root = _required_input_dir(
        bundle_dir,
        '--bundle-dir',
        walks='uploaded every file',
        hint='Pass the bundle dir explicitly — it is the directory `plurai convert-trace` wrote, one folder per scenario.',
    )
    units = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name == BUNDLE_SET_OVERVIEW:
            continue  # set-level overview → inline finalize param, never a lease
        parts = p.relative_to(root).parts
        if any(part.casefold() == CLI_SESSION_DIRNAME.casefold() for part in parts[:-1]):
            continue  # a CLI's recorded session is the local harness's replay input; it never leaves the machine
        if len(parts) == 2 and parts[0].startswith(REGRESSION_SCENARIO_PREFIX) and parts[1] == REGRESSION_REPLAYS_FILENAME:
            continue  # a regression's record-time rate describes the folder; it is not one of its records
        if any(root.joinpath(*parts[: depth + 1]).is_symlink() for depth in range(len(parts))):
            continue  # a link reaches outside what the recorder wrote; only the folder's own files are records
        if p.name == RESUME_PIN_FILENAME:
            _check_pin_is_seedable(p, root)
        if p.name == ISSUE_TURN_FILENAME and (problem := issue_turn_problem(p)):
            raise IngestInputError(
                f'{p.relative_to(root).as_posix()}: {problem}. It is the recorded reply the backend replays against the check '
                'that owns this regression; re-record the folder with /record-regression rather than editing it.'
            )
        units.append(Unit(key=p.relative_to(root).as_posix(), path=p))
    return units


#: The opener a regression of the opener-only shape replays verbatim.
FIRST_TALK_FILENAME = 'first_talk.txt'

#: Every file a recorded regression folder holds, and nothing else: the record step writes the issue
#: paragraph, the recorded issue turn and, for the opener-only shape, the opener; ``convert-trace``
#: adds the pin and its rendering beside a resumed one. A parity check keeps the plugin's recorder to
#: this list, so a file it starts writing is caught instead of quietly putting the folder back under
#: the declaration.
RECORDED_REGRESSION_FILES = frozenset({ROLES_FILENAME, ISSUE_TURN_FILENAME, FIRST_TALK_FILENAME, RESUME_PIN_FILENAME, TRANSCRIPT_FILENAME})


def _pin_resumes_a_seed(folder: Path) -> bool:
    """Whether this folder's pin opens on a resume seed: its converter-stamped ``resume_seed_messages``.

    The run it was cut from was itself resumed, so the pin's leading messages and its state were not
    produced by that run, and nothing in the folder records where the seed came from (a production
    trace seeds one as readily as an earlier regression). An unreadable pin answers yes.
    """
    pin = folder / RESUME_PIN_FILENAME
    if not pin.exists():
        return False
    try:
        provenance = json.loads(pin.read_text(encoding='utf-8')).get('provenance')
    except (OSError, ValueError, AttributeError):
        return True
    return not isinstance(provenance, dict) or 'resume_seed_messages' in provenance


def recorded_regressions_only(bundle_dir: str | Path, units: list[Unit]) -> bool:
    """Whether every record of this bundle is a regression recorded from one of our own simulated runs.

    Such a bundle is run-derived material and uploads under the same default as ``ingest-run``. The
    witness is typed, and every part of it must hold:

    * each record sits directly in a folder (a record at the bundle root, or nested deeper, is in no
      regression) and is one of :data:`RECORDED_REGRESSION_FILES`, a regular file, not a link;
    * each folder's ``issue_turn.json`` is one the backend can replay — only the plugin's recorder
      writes one, from a run transcript;
    * no folder's pin opens on a resume seed, whose origin the folder does not record.

    Anything else is the customer's own data until declared, so it puts the whole bundle back under
    the declaration. ``units`` are :func:`bundle_units`' records, so whatever that excludes is never
    weighed here — except a link: :func:`bundle_units` never uploads one, but a bundle holding one
    still holds something the recorder did not write, so it stays under the declaration.
    """
    root = Path(bundle_dir)
    if any(path.is_symlink() for path in root.rglob('*')):
        return False
    folders: set[str] = set()
    for unit in units:
        folder, sep, name = unit.key.partition('/')
        if not sep or name not in RECORDED_REGRESSION_FILES or unit.path.is_symlink() or unit.path.parent.is_symlink():
            return False
        folders.add(folder)
    return bool(folders) and all(
        issue_turn_problem(root / folder / ISSUE_TURN_FILENAME) is None and not _pin_resumes_a_seed(root / folder) for folder in folders
    )


# --------------------------------------------------------------------------- #
# Publish core (extract already ran; this anonymizes → uploads → manifest)      #
# --------------------------------------------------------------------------- #


def _safe_object_name(key: str, index: int) -> str:
    """A storage-object name (≤64, no path separators) derived from a manifest key."""
    tail = re.sub(r'[^A-Za-z0-9._-]', '_', key)[-40:]
    return f'{index:04d}_{tail}'[:MAX_OBJECT_NAME_LENGTH]  # always ≥5 chars ('NNNN_'), never empty


#: Bounded read chunk for hashing a file (never fully buffered — streams a large member/archive).
_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def _sha256_and_size(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> tuple[str, int]:
    """SHA-256 (hex) + byte length of a file, read in bounded chunks (never fully buffered)."""
    hasher = hashlib.sha256()
    size = 0
    with path.open('rb') as fh:
        while block := fh.read(chunk_bytes):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def _build_archive(units: list[Unit], dest: Path) -> tuple[str, int]:
    """Stream every scrubbed unit into ONE ``.tar.gz`` at ``dest`` (tar member name == ``unit.key``),
    returning the ARCHIVE's SHA-256 + size.

    The bytes depend only on the member names and contents: members are written in sorted name
    order with zero times, owner and mode fields, and the gzip header carries no file name or time.
    The same inputs therefore give the same SHA-256 on every run, which is what lets a backend hold
    a part once and a client skip sending it again.

    STREAMED, never buffered: each unit is copied into the archive by ``tarfile`` block-by-block, and
    the finished archive is hashed in bounded chunks, so neither a single large unit nor the whole
    set is ever held in memory. The tar member name is the unit's manifest ``key`` verbatim (the
    server untars members by that exact name).
    """
    # Duplicate keys would collide as tar members (last-win on untar → silent data loss; the backend also
    # rejects dupes at extraction). Reject them here rather than egress a lossy archive.
    keys = [u.key for u in units]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise IngestInputError(f'duplicate unit keys would collide as tar members (last-win data loss on untar): {dupes}')
    with (
        dest.open('wb') as raw,
        gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=9) as gz,
        tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar,
    ):
        for unit in sorted(units, key=lambda u: u.key):
            info = tarfile.TarInfo(name=unit.key)  # member name == manifest key (the server untars by it)
            info.size = unit.path.stat().st_size
            info.mode = 0o644
            with unit.path.open('rb') as handle:
                tar.addfile(info, handle)  # streams exactly `info.size` bytes from the handle into the archive
    return _sha256_and_size(dest)


def _default_scrub_unit(scrubber: Anonymizer, unit: Unit) -> None:
    """Generic unit scrub: anonymize every string leaf + key of the JSON file in place."""
    scrubber.anonymize_file(unit.path)


def _doc_example_scrub_unit(scrubber: Anonymizer, unit: Unit) -> None:
    """Doc-example unit scrub: the whole file is the example's content (redact-before-egress).

    :meth:`~plurai.ingest.anonymize.Anonymizer.anonymize_bytes` scrubs the file as the document
    itself: UTF-8 text gets the free-text scrub even when it parses as JSON (the generic file scrub
    would re-serialize it), anything else gets the best-effort binary scrub. The scrubbed bytes are
    written back raw, because the archive member must stay the file itself. A refused asset is
    reported under its manifest key, since the unit file's own name is a local counter.
    """
    try:
        scrubbed = scrubber.anonymize_bytes(unit.path.read_bytes(), media_hint=unit.path.suffix)
    except DocExampleBinaryError as e:
        raise DocExampleBinaryError(f'{unit.key}: {e}') from e
    # Atomic swap (tmp + replace), matching Anonymizer.anonymize_file — a crash mid-write must never
    # leave a half-scrubbed unit that could then egress.
    tmp = unit.path.with_name(f'{unit.path.name}.tmp')
    tmp.write_bytes(scrubbed)
    tmp.replace(unit.path)  # atomic in-place swap


def _units_root(units: Sequence[Unit]) -> Path:
    """The directory every unit key is relative to (``<root>/<key>`` is the unit's path)."""
    first = units[0]
    return first.path.parents[len(PurePosixPath(first.key).parts) - 1]


def _stage_bundle_for_upload(units: list[Unit], workspace_root: str | Path, *, staging: Path, scrubber: Anonymizer, err: TextIO) -> list[Unit]:
    """Copy a scenario bundle into ``staging`` as it will leave, folder names pseudonymized
    (anonymized posture only; see :mod:`plurai.ingest.folder_names`).

    The operator's bundle on disk is only READ: the scrub that follows rewrites these staged copies,
    never the originals. The name map is recorded before anything egresses.

    The set-level ``BUNDLE.md`` is sent inline by the start call, not in the archive, so it never
    meets the unit scrub. Its outgoing copy is therefore written EVERY time one exists: folder names
    pseudonymized, then content-scrubbed by the same ``scrubber`` as the archive (the converter
    copies each folder's ``ROLES.md`` opening line into it, which can name a customer), to
    ``OUTGOING_OVERVIEW_RELPATH``. That file, not ``BUNDLE.md``, is what the start call sends.
    """
    root = _units_root(units)
    folders = sorted({parts[0] for unit in units if len(parts := PurePosixPath(unit.key).parts) > 1})
    renames = pseudonyms_for(folders, load_or_create_key(workspace_root))
    staged: list[Unit] = []
    for unit in units:
        parts = PurePosixPath(unit.key).parts
        key = PurePosixPath(renames[parts[0]], *parts[1:]).as_posix() if len(parts) > 1 and parts[0] in renames else unit.key
        dest = staging / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        rewritten = outgoing_text(unit.path, renames) if renames else None
        if rewritten is None:
            shutil.copy2(unit.path, dest)
        else:
            dest.write_text(rewritten, encoding='utf-8')
        staged.append(Unit(key=key, path=dest))
    overview = root / BUNDLE_SET_OVERVIEW
    outgoing_text_ = None
    if overview.is_file():
        with tempfile.TemporaryDirectory(prefix='plurai-ingest-overview-') as overview_dir:  # never an archive member
            staged_overview = Path(overview_dir) / BUNDLE_SET_OVERVIEW
            staged_overview.write_text(outgoing_text(overview, renames) or overview.read_text(encoding='utf-8'), encoding='utf-8')
            scrubber.anonymize_file(staged_overview)
            outgoing_text_ = staged_overview.read_text(encoding='utf-8')
    outgoing = write_outgoing_overview(workspace_root, outgoing_text_)
    if renames:
        print(
            f'plurai: {len(renames)} scenario folder name(s) replaced by keyed pseudonyms for upload '
            f'(anonymize: true); the local map is {record_names(workspace_root, renames)} — `plurai folder-names` reads it.',
            file=err,
        )
    if outgoing is not None:
        print(
            f'plurai: overview_md: send the content of {outgoing} (BUNDLE.md anonymized like the archive), not BUNDLE.md itself.',
            file=err,
        )
    if regressions := [name for name in folders if is_regression_folder(name)]:
        print(
            f'plurai: {len(regressions)} regression folder name(s) sent verbatim: a regression_<slug> is words the recorder '
            'authors by rule (policy + cause word), and they are not screened here.',
            file=err,
        )
    return staged


async def run_ingest(
    *,
    units: list[Unit],
    kind: str,
    workspace_root: str | Path,
    storage_client: StorageClient,
    target: str,
    posture: AnonymizationPosture,
    posture_source: Path | None = None,
    provenance: RunProvenance | None = None,
    scrub_unit: Callable[[Anonymizer, Unit], None] | None = None,
    rename_units: Callable[[Anonymizer, list[Unit]], list[Unit]] | None = None,
    storage_provider_name: str | None = None,
    max_uncompressed_bytes: int | None = None,
    over_cap_remedy: str | None = None,
    err: TextIO | None = None,
) -> IngestManifest:
    """Apply the env's ``posture`` → disclose → bundle into ONE ``.tar.gz`` → upload ONCE → audit →
    return the single-archive :class:`~plurai.ingest.manifest_models.IngestManifest`.

    ``posture`` is required (no default in the egress core — the caller must have resolved the env's
    :func:`~plurai.ingest.posture.resolve_posture`); ``posture_source`` is that resolution's
    PROVENANCE, so the disclosure can tell a declared posture from the unset default instead of
    nagging both alike. Under ``anonymized`` the full scrubber is built
    BEFORE any egress (a missing dep aborts the run with nothing archived/uploaded) and every unit is
    scrubbed IN PLACE first (a scrub raising ⇒ nothing is archived or uploaded); one scrubber is
    shared across units so pseudonyms stay consistent (within-trace/closed-loop). ``rename_units``, when
    given, renames the units under ``anonymized`` (for a kind whose file names carry ids the content
    scrub never sees), and the audit record maps each new key to the unit's own. Under ``raw`` no
    scrubber is built — :meth:`Anonymizer.create` is never called, so Presidio is never imported and
    no ``anon.key`` is minted — and the units are bundled verbatim. Either way the mixed-posture fence
    runs first, then the units are streamed into ONE temporary archive whose SHA-256 + size become the
    manifest. The archive is uploaded once, in parts read from the file (never buffered whole), the
    single-lease audit record is written before leaving the upload block, then the temp archive is
    removed. Disclosure prints to ``err``; the caller prints the returned manifest to stdout.

    After the scrub and before the disclosure, the members' total size is checked against the
    server's total-uncompressed cap (``max_uncompressed_bytes``, else the env override, else the
    backend default; see :func:`~plurai.ingest.archive_cap.resolve_max_uncompressed_bytes`). The server applies that cap
    only when its ``*_start`` verb extracts the archive, after the upload, so an over-cap set raises
    :class:`~plurai.ingest.archive_cap.ArchiveOverCapError` here with nothing archived, uploaded or
    audited. ``over_cap_remedy`` is the command-specific advice the refusal carries.

    ``provenance`` (``ingest-run`` only) is stamped onto the RETURNED manifest and nowhere else: it
    never enters a storage request, so the bytes that egress are the same with it and without it and
    no backend — however old — can fail the upload over a field it does not know.

    **The caller MUST lift the block out and pass its fields as separate arguments.** Handing this
    manifest to ``upload_run_artifacts_start`` verbatim records NOTHING: that tool re-validates
    through a three-field ``extra='ignore'`` model, so ``provenance`` is dropped in silence — no
    error, no warning, and a later replay reports the run as unverifiable. Read the block off this
    output and pass ``eval_set_id`` / ``scenario_set_id`` / ``harness_fingerprint`` /
    ``agent_version`` / ``sdk_version`` / ``sdk_identity`` as the tool's own parameters.

    A docstring in a function the caller never opens cannot guard the drop, so
    :func:`_print_provenance_handoff` also prints the exact call, with the values filled in, on
    stderr beside this manifest at run time.
    """
    stream: TextIO = err if err is not None else sys.stderr
    plan = _UnitPlan(
        kind=kind,
        workspace_root=workspace_root,
        posture=posture,
        scrub_unit=scrub_unit,
        rename_units=rename_units,
        max_uncompressed_bytes=max_uncompressed_bytes,
        over_cap_remedy=over_cap_remedy,
    )
    with _prepared_units(units, plan, err=stream) as prepared:
        scrubber = prepared.scrubber
        print(_upload_preview(prepared, plan, unit_count=len(prepared.units), target=target, posture_source=posture_source).render(), file=stream)

        # Bundle ALL scrubbed units into ONE streamed .tar.gz and upload it once. A TemporaryDirectory
        # scopes the archive so it is always cleaned up (success or error). A failure raises with NO
        # manifest returned (fail-closed — nothing is audited that did not actually leave).
        with tempfile.TemporaryDirectory(prefix='plurai-ingest-') as tmpdir:
            archive_path = Path(tmpdir) / 'ingest.tar.gz'
            archive_checksum, archive_bytes = _build_archive(prepared.units, archive_path)
            lease_id = (
                await storage_client.upload_archive(
                    archive_path,
                    name=_safe_object_name(f'{kind}-archive', 0),
                    checksum=archive_checksum,
                    content_length=archive_bytes,
                    content_type='application/gzip',
                    storage_provider_name=storage_provider_name,
                )
            ).lease_id
            # The archive has now egressed (→ ONE server-minted lease). Write the compliance audit record
            # IMMEDIATELY, still inside the upload block — so bytes can never leave without a local record
            # of exactly what left. A failed audit write raises IngestAuditError (naming the lease /
            # checksum / bytes to record by hand), never a silent drop. The member keys are logged for the
            # disclosure trail (what files were bundled) — not per-member checksums/sizes (the archive
            # checksum already pins them).
            write_audit_entry(
                workspace_root,
                kind=kind,
                lease_id=lease_id,
                checksum=archive_checksum,
                archive_bytes=archive_bytes,
                member_keys=[unit.key for unit in prepared.units],
                member_sources=prepared.member_sources,
                posture=posture,
                key_fingerprint=scrubber.key_fingerprint if scrubber is not None else None,
                scrub_summary=scrubber.scrub_summary if scrubber is not None else None,
            )
        return IngestManifest(
            lease_id=UUID(lease_id),
            checksum=archive_checksum,
            bytes=archive_bytes,
            posture=posture,
            anon_key_fingerprint=scrubber.key_fingerprint if scrubber is not None else None,
            provenance=provenance,
        )


@dataclass(frozen=True)
class _UnitPlan:
    """What an ingest does to its units before anything is disclosed, and the limits it holds them to."""

    kind: str
    workspace_root: str | Path
    posture: AnonymizationPosture
    scrub_unit: Callable[[Anonymizer, Unit], None] | None
    rename_units: Callable[[Anonymizer, list[Unit]], list[Unit]] | None
    max_uncompressed_bytes: int | None
    over_cap_remedy: str | None


@dataclass(frozen=True)
class _Prepared:
    """Units after the posture is applied: scrubbed (and renamed) under ``anonymized``, verbatim under ``raw``."""

    units: list[Unit]
    scrubber: Anonymizer | None
    member_sources: dict[str, str] | None
    total_bytes: int


@contextlib.contextmanager
def _prepared_units(units: list[Unit], plan: _UnitPlan, *, err: TextIO) -> Iterator[_Prepared]:
    """Everything an ingest does to its units before anything is disclosed.

    The mixed-posture fence runs first. Under ``anonymized`` the full scrubber is built here (fail-closed:
    a missing presidio/spaCy dep raises before any egress) and every unit is scrubbed IN PLACE, for a
    scenario bundle in a staged copy that is removed when the context exits. Under ``raw`` no scrubber is
    built, so Presidio is never imported and no ``anon.key`` is minted. The members' total size is then
    checked against the server's total-uncompressed cap, and ``rename_units`` (anonymized only) renames
    units whose file names carry ids the content scrub never sees.
    """
    kind, workspace_root, posture = plan.kind, plan.workspace_root, plan.posture
    if not units:
        raise IngestInputError(f'nothing to ingest for {kind!r}: 0 units')

    # Posture-consistency fence BEFORE any dep check / scrub / egress: refuse to add a
    # differently-postured upload to an env that already shipped one.
    assert_posture_consistent(workspace_root, posture, kind=kind)
    # Defence in depth behind `_dispatch`'s raw stamp: a run kind is the simulation's own output and
    # is never scrubbed, so a caller that reaches this with a run kind under any other posture is refused.
    if kind in RUN_KINDS and posture != 'raw':
        raise ValueError(
            f'{kind!r} is a run kind and uploads raw whatever the anonymization posture '
            f'(the posture governs only the env-build uploads); got posture {posture!r}'
        )

    with contextlib.ExitStack() as staging_scope:
        scrubber: Anonymizer | None = None
        member_sources: dict[str, str] | None = None
        if posture == 'anonymized':
            scrubber = Anonymizer.create(workspace_root)
            if kind in PSEUDONYMIZED_FOLDER_KINDS:
                # Staged before the scrub: the scrub then rewrites copies, and never sees (or half-hashes) a
                # folder name it was not asked to keep.
                staging = Path(staging_scope.enter_context(tempfile.TemporaryDirectory(prefix='plurai-ingest-bundle-')))
                units = _stage_bundle_for_upload(units, workspace_root, staging=staging, scrubber=scrubber, err=err)
            scrub = plan.scrub_unit if plan.scrub_unit is not None else _default_scrub_unit
            for unit in units:
                if not unit.verbatim:
                    scrub(scrubber, unit)  # scrub before the bytes can leave (raises → nothing archived/uploaded)

        # Sized after the scrub, which rewrites the files: these are the tar member sizes the server sums.
        total_bytes = check_uncompressed_total(
            [(unit.key, unit.path.stat().st_size) for unit in units],
            resolve_max_uncompressed_bytes(plan.max_uncompressed_bytes),
            remedy=plan.over_cap_remedy or _DEFAULT_OVER_CAP_REMEDY,
        )
        # Renamed after the cap check, so an over-cap refusal names the operator's own files.
        if scrubber is not None and plan.rename_units is not None:
            renamed = plan.rename_units(scrubber, units)
            member_sources = {new.key: old.key for new, old in zip(renamed, units, strict=True)}
            units = renamed
        yield _Prepared(units=units, scrubber=scrubber, member_sources=member_sources, total_bytes=total_bytes)


def _upload_preview(
    prepared: _Prepared, plan: _UnitPlan, *, unit_count: int, target: str, posture_source: Path | None, source: str | None = None
) -> UploadPreview:
    """The pre-upload disclosure of prepared units: what is sent, where, under which posture and key."""
    scrubber = prepared.scrubber
    return UploadPreview(
        kind=plan.kind,
        unit_count=unit_count,
        total_bytes=prepared.total_bytes,
        target=target,
        posture=plan.posture,
        scrub_summary=scrubber.scrub_summary if scrubber is not None else None,
        key_fingerprint=scrubber.key_fingerprint if scrubber is not None else None,
        posture_source=posture_source,
        source=source,
    )


@dataclass(frozen=True)
class PartScrub:
    """How an anonymized part is scrubbed, and the recorder that scrub fills and the part's scrub map is written from.

    The two travel as one value so the map always describes the scrub that ran.
    """

    unit: Callable[[Anonymizer, Unit], None]
    recorder: ScrubRecorder


async def run_part_ingest(
    *,
    units: list[Unit],
    kind: str,
    part_kind: PartKind,
    workspace_root: str | Path,
    storage_client: StorageClient | None,
    target: str | None,
    posture: AnonymizationPosture,
    label: str | None = None,
    posture_source: Path | None = None,
    scrub: PartScrub | None = None,
    rename_units: Callable[[Anonymizer, list[Unit]], list[Unit]] | None = None,
    max_uncompressed_bytes: int | None = None,
    over_cap_remedy: str | None = None,
    expect_sha256: str | None = None,
    source: str | None = None,
    err: TextIO | None = None,
) -> PartEntry:
    """Build ONE content-addressed evidence part from ``units``, disclose it, upload it if the backend lacks it,
    and return the entry the plugin passes to ``ground_workspace_start``.

    ``expect_sha256`` is the hash an approved plan named: a part that hashes differently raises
    :class:`~plurai.ingest.parts.PartMismatchError` before anything is disclosed or sent. Without a
    ``storage_client`` (and ``target``) the part is only planned: it is built exactly as for an upload, the
    entry is returned and one ``planned`` audit entry is written, and no request is possible.

    The units meet the posture as in :func:`run_ingest`. An ``anonymized`` part also carries ``scrub_map.json``
    as an archive member, listing what ``scrub.recorder`` recorded while ``scrub.unit`` ran (no ``scrub`` means the
    generic scrub and an empty map); the member is part of the bytes the hash covers. The disclosure (every part
    with its size and posture), the disclosure file and the audit entry for the part are all written before the
    first request that carries content; the by-hash lookup that decides whether to send anything comes after.
    """
    if (storage_client is None) != (target is None):
        raise ValueError('a part is either planned (no storage client, no target) or delivered (both)')
    stream: TextIO = err if err is not None else sys.stderr
    plan = _UnitPlan(
        kind=kind,
        workspace_root=workspace_root,
        posture=posture,
        scrub_unit=scrub.unit if scrub is not None else None,
        rename_units=rename_units,
        max_uncompressed_bytes=max_uncompressed_bytes,
        over_cap_remedy=over_cap_remedy,
    )
    with _prepared_units(units, plan, err=stream) as prepared:
        scrubber = prepared.scrubber
        members = list(prepared.units)
        with tempfile.TemporaryDirectory(prefix='plurai-part-') as tmpdir:
            if scrubber is not None:
                map_path = Path(tmpdir) / SCRUB_MAP_MEMBER
                recorder = scrub.recorder if scrub is not None else ScrubRecorder()
                map_path.write_bytes(recorder.member_bytes())
                members.append(Unit(key=SCRUB_MAP_MEMBER, path=map_path))
            archive_path = Path(tmpdir) / 'part.tar.gz'
            archive_hex, archive_bytes = _build_archive(members, archive_path)
            sha256 = f'sha256:{archive_hex}'
            entry = PartEntry(
                kind=part_kind,
                label=label,
                sha256=sha256,
                bytes=archive_bytes,
                posture=posture,
                anon_key_fingerprint=scrubber.key_fingerprint if scrubber is not None else None,
            )
            part = BuiltPart(
                entry=entry,
                path=archive_path,
                object_name=_safe_object_name(f'{part_kind}-part', 0),
                content_type='application/gzip',
                members=tuple(unit.key for unit in members),
                verbatim=tuple(unit.key for unit in members if unit.verbatim),
                member_sources=prepared.member_sources,
                scrub_summary=scrubber.scrub_summary if scrubber is not None else None,
            )
            if expect_sha256 is not None:
                require_expected_sha256(entry, expect_sha256)
            if storage_client is None or target is None:
                plan_parts([part], workspace_root=workspace_root, kind=kind)
                return entry
            preview = _upload_preview(prepared, plan, unit_count=len(members), target=target, posture_source=posture_source, source=source)
            disclosure = disclose_parts([part], workspace_root=workspace_root, kind=kind, text='\n'.join([preview.render(), *part_lines([part])]), err=stream)
            await upload_disclosed(storage_client, disclosure)
        return entry


#: The largest agent description sent; the backend refuses a larger one, so it is refused before it is sent.
MAX_DESCRIPTION_BYTES = 8 * 1024 * 1024


async def run_descriptions_ingest(
    *,
    files: list[tuple[str, Path]],
    workspace_root: str | Path,
    storage_client: StorageClient | None,
    target: str | None,
    expect_sha256: Mapping[str, str] | None = None,
    err: TextIO | None = None,
) -> list[PartEntry]:
    """Deliver agent descriptions as ``agent_description`` parts named by file, and return their entries.

    A description is sent as written: it is the customer's own agent description, carries no posture and is
    not scrubbed. Each file is copied once into a staging directory and everything after that, the hash, the
    disclosure and the upload, reads the copy, so the bytes disclosed are the bytes sent. A file that is not
    UTF-8 text, or is larger than :data:`MAX_DESCRIPTION_BYTES`, is refused before anything is disclosed.

    ``expect_sha256`` maps part names to the hashes an approved plan named; the parts built must be exactly those,
    or :class:`~plurai.ingest.parts.PartMismatchError` is raised before anything is disclosed or sent. Without a
    ``storage_client`` (and ``target``) the parts are only planned: their entries are returned and one ``planned``
    audit entry each is written, and no request is possible.
    """
    if (storage_client is None) != (target is None):
        raise ValueError('descriptions are either planned (no storage client, no target) or delivered (both)')
    stream: TextIO = err if err is not None else sys.stderr
    names = [name for name, _ in files]
    if not files:
        raise IngestInputError('no agent description to deliver: none of the description files was found')
    if len(set(names)) != len(names):
        raise IngestInputError(f'two agent descriptions share one name: {sorted({n for n in names if names.count(n) > 1})}')
    with tempfile.TemporaryDirectory(prefix='plurai-descriptions-') as tmpdir:
        parts: list[BuiltPart] = []
        for name, source in files:
            raw = source.read_bytes()
            if not raw:
                raise IngestInputError(f'{source} is empty; an agent description has at least one byte')
            if len(raw) > MAX_DESCRIPTION_BYTES:
                raise IngestInputError(f'{source} is {len(raw)} bytes; an agent description is at most {MAX_DESCRIPTION_BYTES}')
            try:
                raw.decode('utf-8')
            except UnicodeDecodeError:
                raise IngestInputError(f'{source} is not UTF-8 text; an agent description is a text file') from None
            staged = Path(tmpdir) / f'{len(parts):02d}'
            staged.write_bytes(raw)
            sha256, size = sha256_of(staged)
            parts.append(
                BuiltPart(
                    entry=PartEntry(kind='agent_description', name=name, sha256=sha256, bytes=size),
                    path=staged,
                    object_name=_safe_object_name('agent-description', len(parts)),
                    content_type='application/json' if name.endswith('.json') else 'text/plain; charset=utf-8',
                    members=(name,),
                )
            )
        entries = [part.entry for part in parts]
        if expect_sha256 is not None:
            require_expected_sha256_by_name(entries, expect_sha256)
        if storage_client is None or target is None:
            plan_parts(parts, workspace_root=workspace_root, kind='agent-descriptions')
            return entries
        disclosure = disclose_parts(
            parts, workspace_root=workspace_root, kind='agent-descriptions', text=render_parts_disclosure(parts, target=target), err=stream
        )
        await upload_disclosed(storage_client, disclosure)
    return entries


def description_files(directory: str | Path, *, scan_contract: str | Path | None = None, err: TextIO) -> list[tuple[str, Path]]:
    """The description files to deliver, ``(part name, path)``: the ones present in ``directory``, then the scan
    contract when given."""
    found: list[tuple[str, Path]] = []
    for name in DESCRIPTION_FILES:
        path = Path(directory) / name
        if path.is_file():
            found.append((name, path))
        else:
            print(f'plurai ingest-descriptions: {path} not found; it is not delivered.', file=err)
    if scan_contract is not None:
        if not Path(scan_contract).is_file():
            raise IngestInputError(f'--scan-contract {scan_contract} is not a file')
        found.append((SCAN_CONTRACT_NAME, Path(scan_contract)))
    return found


@dataclass(frozen=True)
class PartSpec:
    """A grounding part an ingest command builds in place of a lease: its kind, its label, and how it is scrubbed."""

    kind: PartKind
    label: str | None = None
    scrub: PartScrub | None = None
    # Where the units came from when that is a local folder; the disclosure names it.
    source: str | None = None


def _trace_part_scrub(declared: frozenset[str]) -> PartScrub:
    """The scrub of a trace part: declared names keep their spelling and every rewritten tool field is recorded."""
    recorder = ScrubRecorder()
    return PartScrub(_declared_scrub_unit(declared, recorder), recorder)


def _corpus_part_scrub(declared: frozenset[str], claims: tuple[Claim, ...]) -> PartScrub:
    """The scrub of a corpus part: declared names keep their spelling, and a scan claim whose matched text the scrub
    rewrote is recorded, so the part's scrub map names the claim."""
    recorder = ScrubRecorder()
    unit = _declared_scrub_unit(declared, recorder) if declared else _default_scrub_unit
    return PartScrub(claim_scrub_unit(unit, claims, recorder) if claims else unit, recorder)


def _declared_scrub_unit(declared: frozenset[str], recorder: ScrubRecorder) -> Callable[[Anonymizer, Unit], None]:
    """A unit scrub that keeps declared names as they are and records which tool fields it rewrote.

    A unit that is not JSON gets the generic scrub, which also fails closed on a binary.
    """

    def scrub(scrubber: Anonymizer, unit: Unit) -> None:
        try:
            data = json.loads(unit.path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeDecodeError):
            _default_scrub_unit(scrubber, unit)
            return
        scrubbed = scrub_trace_tree(data, scrubber, declared, recorder)
        tmp = unit.path.with_name(f'{unit.path.name}.tmp')
        tmp.write_text(json.dumps(scrubbed, ensure_ascii=False), encoding='utf-8')
        tmp.replace(unit.path)  # atomic in-place swap

    return scrub


# --------------------------------------------------------------------------- #
# argparse wiring                                                              #
# --------------------------------------------------------------------------- #


def _dispatch(
    args: argparse.Namespace,
    *,
    units: list[Unit],
    kind: str,
    declaration_required: bool,
    scrub_unit: Callable[[Anonymizer, Unit], None] | None = None,
    rename_units: Callable[[Anonymizer, list[Unit]], list[Unit]] | None = None,
    provenance: RunProvenance | None = None,
    over_cap_remedy: str | None = None,
    part: PartSpec | None = None,
) -> int:
    plan_only = part is not None and bool(getattr(args, 'plan_only', False))
    # Every upload kind, before anything leaves: what it names of the operator's home (a warning, never a stop).
    _warn_home_named(units, args.workspace_root, command=getattr(args, 'command', None) or f'ingest-{kind}')
    # Resolve the env's anonymization posture FIRST, so a config error (unparseable /
    # non-bool / conflicting plurai.config.yaml) fails closed BEFORE any storage client is built.
    # (The local extractor in _cmd_* has already run by now, so a malformed config costs
    # local-only extraction work — never egress.)
    #
    # `declaration_required` is stated by every caller: True when the units carry the customer's own
    # data out of the machine, which uploads only under a DECLARED posture (an absent `anonymize:` is
    # nobody's answer, so it is refused rather than read as raw); False for material our own simulated
    # runs produced, which keeps the default.
    #
    # The posture governs only what is uploaded while the env is built: a run kind's units are the
    # simulation's own output and upload as-is, so the config is not read for them.
    resolution = PostureResolution(posture='raw', source=None) if kind in RUN_KINDS else resolve_posture(args.workspace_root)
    if resolution.source is None and declaration_required:
        raise PostureConfigError(
            f'no `{POSTURE_CONFIG_KEY}:` declaration under {args.workspace_root}, and {kind} carries customer data. '
            f'Declare `{POSTURE_CONFIG_KEY}: true` (scrub locally before upload) or `{POSTURE_CONFIG_KEY}: false` (upload as-is) '
            'in plurai.config.yaml, then re-run. Nothing was uploaded.'
        )

    # Resolved here, not left to run_ingest, so a malformed cap override is refused before the
    # storage client or the scrubber is built.
    max_uncompressed_bytes = resolve_max_uncompressed_bytes()

    # The storage client is built from the configured backend (URL + bearer); the disclosure
    # target is read OFF THE CLIENT (client.base_url) — the consent surface shows the
    # exact base the upload hits, so disclosure and egress cannot drift.
    async def _go() -> IngestManifest | PartEntry:
        if part is not None and plan_only:
            # No client is built, so a plan needs no backend credentials and has no way to reach the network.
            return await run_part_ingest(
                units=units,
                kind=kind,
                part_kind=part.kind,
                label=part.label,
                workspace_root=args.workspace_root,
                storage_client=None,
                target=None,
                posture=resolution.posture,
                posture_source=resolution.source,
                scrub=part.scrub,
                rename_units=rename_units,
                max_uncompressed_bytes=max_uncompressed_bytes,
                over_cap_remedy=over_cap_remedy,
                expect_sha256=getattr(args, 'expect_sha256', None),
            )
        async with StorageClient.from_env() as client:
            if part is not None:
                return await run_part_ingest(
                    units=units,
                    kind=kind,
                    part_kind=part.kind,
                    label=part.label,
                    workspace_root=args.workspace_root,
                    storage_client=client,
                    target=client.base_url,
                    posture=resolution.posture,
                    posture_source=resolution.source,
                    scrub=part.scrub,
                    rename_units=rename_units,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                    over_cap_remedy=over_cap_remedy,
                    expect_sha256=getattr(args, 'expect_sha256', None),
                    source=part.source,
                )
            return await run_ingest(
                units=units,
                kind=kind,
                workspace_root=args.workspace_root,
                storage_client=client,
                target=client.base_url,
                posture=resolution.posture,
                posture_source=resolution.source,
                provenance=provenance,
                scrub_unit=scrub_unit,
                rename_units=rename_units,
                max_uncompressed_bytes=max_uncompressed_bytes,
                over_cap_remedy=over_cap_remedy,
            )

    manifest = asyncio.run(_go())
    # stdout = the part entry (a grounding part) or the single-archive manifest (a lease), for the plugin
    shown = planned_entry(manifest) if plan_only and isinstance(manifest, PartEntry) else manifest.model_dump(mode='json')
    print(json.dumps(shown, ensure_ascii=False))
    if plan_only:
        print(f'plurai {getattr(args, "command", None) or f"ingest-{kind}"}: plan only; nothing was sent.', file=sys.stderr)
    if isinstance(manifest, IngestManifest):
        _print_provenance_handoff(manifest, err=sys.stderr)
    return 0


#: The public MCP tool of the Plurai plugin that this CLI's ``run-artifacts`` manifest is handed to, and the parameter names it
#: takes the provenance under. Named here rather than only in prose so the hand-off notice
#: cannot drift from the call it describes.
_RUN_UPLOAD_TOOL = 'upload_run_artifacts_start'
#: The provenance keys that tool DECLARES as parameters. Its MCP server ignores an undeclared
#: argument, so a stamped key outside this tuple is named as manifest-only and never offered as an
#: argument (nor its value printed). ``sdk_identity`` is a dict, printed as JSON; the rest are strings.
_RUN_UPLOAD_PARAMS: tuple[str, ...] = ('eval_set_id', 'scenario_set_id', 'harness_fingerprint', 'agent_version', 'sdk_version', 'sdk_identity')


def _print_provenance_handoff(manifest: IngestManifest, *, err: TextIO) -> None:
    """Print the EXACT tool call this manifest's provenance has to be handed over as.

    THE WIRE SHAPE IS SPLIT, and this is the guard on that split. The CLI stamps provenance
    NESTED inside the manifest (where the backend's manifest model — three fields, unknown
    keys ignored — silently drops it), while ``upload_run_artifacts_start`` takes it only as
    separate PARAMETERS; the backend does not read the nested block. So the natural move, handing
    the printed manifest over verbatim, records nothing: no error, no warning, and a later replay
    reports the run as unverifiable for reasons nobody can see. The drop is guarded HERE, at the
    moment the values exist and in front of the operator who has them — a docstring inside a
    function nobody opens is not a guard.

    Prints nothing when there is no provenance to hand over: every non-run ingest kind, and a
    run whose caller stamped no ids at all.
    """
    if manifest.provenance is None:
        return
    stamped = manifest.provenance.model_dump(mode='json')  # absent keys are already omitted
    passed = {key: value for key, value in stamped.items() if key in _RUN_UPLOAD_PARAMS}
    manifest_only = [key for key in stamped if key not in _RUN_UPLOAD_PARAMS]
    args = ', '.join(f'{key}={json.dumps(value, sort_keys=True)}' if isinstance(value, dict) else f'{key}="{value}"' for key, value in passed.items())
    only = f'  Manifest only — {_RUN_UPLOAD_TOOL} takes no parameter for: {", ".join(manifest_only)}.\n' if manifest_only else ''
    print(
        f'[provenance] {len(passed)} field(s) to pass. They are NESTED in the manifest above and '
        f'{_RUN_UPLOAD_TOOL} DROPS them there (its manifest model ignores unknown keys) — pass them as its OWN '
        f'arguments:\n'
        f'  {_RUN_UPLOAD_TOOL}(manifest=<the JSON above>, {args})\n'
        f'{only}'
        f'  Omit them and score_run can only warn that it does not know whether these artifacts are stale.',
        file=err,
    )


def _json_object(raw: str, flag: str) -> dict:
    """Parse an inline-JSON CLI value, insisting it is an object."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise IngestInputError(f'{flag} must be valid JSON: {e}') from e
    if not isinstance(value, dict):
        raise IngestInputError(f'{flag} must be a JSON object, got {type(value).__name__}.')
    return value


def _corpus_load_kwargs(args: argparse.Namespace) -> dict:
    """Chunking overrides for ``source.load()`` — ONLY the flags actually passed.

    Absent flags must land on the SOURCE's own defaults (section chunking for markdown),
    not on a copy of them restated here: a default duplicated in the CLI is a default that
    silently stops tracking the extractor the day the extractor's changes.
    """
    overrides = {'chunk_size': args.chunk_size, 'chunk_overlap': args.chunk_overlap, 'chunk_mode': args.chunk_mode}
    return {k: v for k, v in overrides.items() if v is not None}


def _read_scan_contract(path: str | Path) -> dict[str, Any]:
    """The scan's ``scan_contract.json`` as a JSON object."""
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise IngestInputError(f'--scan-contract {path}: not readable as JSON ({e}).') from e
    if not isinstance(payload, dict):
        raise IngestInputError(f'--scan-contract {path}: expected a JSON object, got {type(payload).__name__}.')
    return payload


def _corpus_scope(args: argparse.Namespace) -> tuple[str, ...] | None:
    """The suffix scope the scan read a ``directory`` source with — a REQUIRED input for one, ``None`` for any other kind.

    From ``--scan-contract`` (the scan's own ``scan_contract.json``: its recorded
    ``source_inventory.suffixes``), or from ``--suffixes`` spelled exactly as the scan's own flag
    (the EXTRA suffixes, on top of its default set; the flag with no value is the default set).
    """
    from plurai.scan_contract.corpus import SourceInventory
    from plurai.scan_contract_cli import admitted_suffixes

    given = [flag for flag, value in (('--scan-contract', args.scan_contract), ('--suffixes', args.suffixes)) if value is not None]
    if args.source_kind != 'directory':
        if given:
            raise IngestInputError(f'{" and ".join(given)} appl{"y" if len(given) > 1 else "ies"} to a directory source, not to {args.source_kind!r}.')
        return None
    if len(given) > 1:
        raise IngestInputError("pass the scan scope once: --scan-contract (the scan's recorded scope) OR --suffixes, not both.")
    if args.suffixes is not None:
        return admitted_suffixes(args.suffixes)
    if args.scan_contract is None:
        raise IngestInputError(
            'a directory source needs the suffix scope the scan read it with, so the corpus and the scan describe the same files: '
            'pass --scan-contract <plurai_sim/artifacts/scan_contract.json>, or --suffixes with the extra suffixes the scan was run '
            'with (--suffixes alone for its default set).'
        )
    payload = _read_scan_contract(args.scan_contract)
    inventory = payload.get('source_inventory')
    recorded = SourceInventory.from_dict(inventory).suffixes if isinstance(inventory, dict) else ()
    if not recorded:
        raise IngestInputError(
            f'--scan-contract {args.scan_contract} records no source_inventory.suffixes, so the scope the scan read is unknown — '
            're-run the scan, or pass --suffixes with the extra suffixes it was run with.'
        )
    return recorded


def _cmd_corpus(args: argparse.Namespace) -> int:
    from plurai.rag_extract import extract_corpus

    if args.plan_only and args.extract_only:
        raise IngestInputError('--plan-only builds the part and --extract-only stops before it exists; give one of them')
    if args.domain_file is not None:
        read_domain_file(args.domain_file)  # refused before any extraction work
    # Read before the extraction too: an unreadable declaration is refused before the source is exported, not after.
    declared = load_declared_names(args.declared_names) if args.declared_names and not args.extract_only else frozenset()
    claims = load_claims(_read_scan_contract(args.scan_contract)) if args.scan_contract is not None and not args.extract_only else ()
    source_config = json.loads(Path(args.source_config).read_text(encoding='utf-8')) if args.source_config else {}
    manifest = extract_corpus(
        args.source_kind,
        source_config,
        args.out_dir,
        domain=_json_object(args.domain, '--domain') if args.domain else None,
        load_kwargs=_corpus_load_kwargs(args),
        roots=args.root,
        suffixes=_corpus_scope(args),
    )
    for warning in manifest.get('skipped_suffix_warnings') or []:
        print(f'plurai ingest-corpus: WARNING: {warning}', file=sys.stderr)
    not_ingested = manifest.get('not_ingested') or []
    for dropped in not_ingested:
        print(f'plurai ingest-corpus: not ingested: {dropped["path"]} ({dropped["reason"]})', file=sys.stderr)
    for part in manifest.get('partially_ingested') or []:
        pages = ', '.join(str(n) for n in part['pages'])
        print(f'plurai ingest-corpus: partly ingested: {part["path"]} pages {pages} of {part["of"]} ({part["reason"]})', file=sys.stderr)
    if args.extract_only:
        # Return BEFORE _dispatch: no posture resolution, no scrub, no archive, no lease, no upload,
        # no audit entry. An under-chunked corpus otherwise costs a full upload + backend build to see.
        summary = corpus_extraction_summary(args.out_dir, args.source_kind)
        staged = manifest.get('staged_files')
        # The files that reached a chunk, counted from the chunks: a glob narrower than the staging leaves a
        # file neither read nor named, so staged minus not_ingested would count it as read.
        read_from = f'{summary["source_documents"]} of {staged} staged file(s)' if staged is not None else f'{summary["source_documents"]} source document(s)'
        print(
            f'plurai ingest-corpus: extract-only — {summary["chunks"]} chunk(s) from {read_from}, '
            f'{summary["relationships"]} relationship(s) ({summary["relationships_per_chunk"]} per chunk that owes one) in {args.out_dir}; nothing uploaded.',
            file=sys.stderr,
        )
        per_document = summary['per_document']
        for name, count in per_document.items() if isinstance(per_document, dict) else ():
            print(f'  {name}: {count} chunk(s)', file=sys.stderr)
        # A SECOND stderr line, because the numbers alone mean different things per source kind.
        numbers = {key: value for key, value in summary.items() if not isinstance(value, dict)}
        print(f'plurai ingest-corpus: {corpus_shape_note(numbers, args.source_kind)}', file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False))  # stdout = the summary, in the manifest's place
        return 0
    units = corpus_units(args.out_dir)
    if args.domain_file is not None:
        units.append(domain_unit(args.domain_file, args.out_dir))
    return _dispatch(
        args,
        units=units,
        kind='corpus',
        declaration_required=True,
        part=PartSpec('corpus_source', label=args.label, scrub=_corpus_part_scrub(declared, claims)),
    )


def _cmd_descriptions(args: argparse.Namespace) -> int:
    files = description_files(args.descriptions_dir or args.workspace_root, scan_contract=args.scan_contract, err=sys.stderr)

    expected = _expected_by_name(args.expect_sha256)

    async def _go() -> list[PartEntry]:
        if args.plan_only:
            return await run_descriptions_ingest(files=files, workspace_root=args.workspace_root, storage_client=None, target=None, expect_sha256=expected)
        async with StorageClient.from_env() as client:
            return await run_descriptions_ingest(
                files=files, workspace_root=args.workspace_root, storage_client=client, target=client.base_url, expect_sha256=expected
            )

    entries = asyncio.run(_go())
    # stdout = the part entries, in delivery order
    print(json.dumps([planned_entry(entry) if args.plan_only else entry.model_dump(mode='json') for entry in entries], ensure_ascii=False))
    if args.plan_only:
        print('plurai ingest-descriptions: plan only; nothing was sent.', file=sys.stderr)
    return 0


def _expected_by_name(pairs: list[tuple[str, str]] | None) -> dict[str, str] | None:
    """The ``--expect-sha256 NAME=HASH`` pairs as a map, or ``None`` when no hash was given. A name given twice is refused."""
    if not pairs:
        return None
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise IngestInputError(f'--expect-sha256 names {sorted({n for n in names if names.count(n) > 1})} more than once')
    return dict(pairs)


def _cmd_traces(args: argparse.Namespace) -> int:
    from plurai.trace_extract import extract_traces

    declared = load_declared_names(args.declared_names) if args.declared_names else frozenset()  # refused before the sink is exported
    source_config = json.loads(Path(args.source_config).read_text(encoding='utf-8')) if args.source_config else {}
    manifest = extract_traces(args.source_kind, source_config, args.out_dir, limit=args.limit)
    return _dispatch(
        args,
        units=trace_units(args.out_dir),
        kind='traces',
        declaration_required=True,
        rename_units=pseudonymize_trace_units,
        over_cap_remedy=_trace_over_cap_remedy(args.source_kind),
        part=PartSpec('trace_archive', scrub=_trace_part_scrub(declared), source=_trace_source_line(args.source_kind, manifest)),
    )


def _trace_source_line(source_kind: str, manifest: dict[str, Any]) -> str | None:
    """The disclosure's source line for a trace corpus read from a folder: ``export_dir <path>``. A live sink has none."""
    if source_kind != 'export_dir':
        return None
    return f'export_dir {manifest["selection"]["export_dir"]["path"]}'


def _trace_over_cap_remedy(source_kind: str) -> str:
    """How this trace source exports fewer traces, for the over-cap refusal."""
    keys = trace_selection_keys().get(source_kind, [])
    narrowing = f'select fewer traces with {" or ".join(f"`{key}`" for key in keys)} in --source-config (leave out the largest above), or ' if keys else ''
    return f'Narrow the export and re-run: {narrowing}lower --limit.'


def _cmd_doc_examples(args: argparse.Namespace) -> int:
    from plurai.doc_examples_extract import extract_doc_examples

    # The third member of the class: the extractor ITERATES ``--source-dir``, so an empty value
    # scans the cwd for family folders. Refused here rather than inside the extractor, beside
    # the other two, so the rule has one reader.
    _required_input_dir(
        args.source_dir,
        '--source-dir',
        walks='scanned every subdirectory for document-example families',
        hint='Pass the folder that holds one subdirectory per example family, each with its FAMILY.md.',
    )
    extract_doc_examples(args.source_dir, args.out_dir)
    return _dispatch(
        args,
        units=doc_example_units(args.out_dir),
        kind='doc-examples',
        declaration_required=True,
        scrub_unit=_doc_example_scrub_unit,  # reach inside content_b64 (redact-before-egress)
    )


def _is_utf8(path: Path) -> bool:
    try:
        path.read_bytes().decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def _twin_text(path: Path, *, err: TextIO) -> readers.TextRead:
    """One record's text through :mod:`plurai.readers`: a file by its format, a zip as its members' texts in order."""
    if not path.name.lower().endswith(readers.ZIP_SUFFIX):
        return readers.to_text(path.name, path.read_bytes())
    sections: list[str] = []
    skipped: list[str] = []
    for member, raw in readers.members(path):
        read = readers.to_text(f'{path.name}{member}', raw)
        if read.text is None:
            skipped.append(f'{path.name}{member} ({read.reason})')
        else:
            sections.append(f'# {path.name}{member}\n\n{read.text}')
    if skipped:
        print(f'plurai ingest-bundle: not in the text copy of {path.name}: {"; ".join(skipped)}.', file=err)
    return readers.TextRead('\n\n'.join(sections)) if sections else readers.TextRead(None, 'no member has text')


def bundle_text_twins(units: Sequence[Unit], stage: Path, *, err: TextIO) -> list[Unit]:
    """A UTF-8 text copy, keyed ``<key>.md``, of each record the backend could not read as text.

    The backend grounds a scenario in its records' UTF-8 text, so a converted format (``.docx``, ``.pdf``,
    ``.html`` …), a gzip or a zip reaches it only through this copy. Copies are written under ``stage``,
    never into the bundle. Uploaded beside their binary, they follow every unit's path: under
    ``anonymized`` they get the same free-text scrub as any text record, and the binary itself keeps its
    own scrub (an office file or an archive is refused there). A record with no text — no
    reader, or one that yields none — gets no copy and one line on ``err`` naming it.
    """
    keys = {unit.key for unit in units}
    twins: list[Unit] = []
    for unit in units:
        name = unit.path.name.lower()
        kind = readers.kind_of(name)
        if not name.endswith((readers.GZIP_SUFFIX, readers.ZIP_SUFFIX)):
            if kind in ('text', 'records'):
                continue  # stored as UTF-8 text already
            if kind is None and _is_utf8(unit.path):
                continue  # no reader, but already text the backend reads
        key = f'{unit.key}{readers.TEXT_TWIN_SUFFIX}'
        if key in keys:
            print(f'plurai ingest-bundle: {unit.key}: no text copy written, the bundle already holds {key}.', file=err)
            continue
        read = _twin_text(unit.path, err=err)
        if read.text is None:
            print(f'plurai ingest-bundle: {unit.key}: no text copy ({read.reason}); the backend grounds on text, so it will not read this file.', file=err)
            continue
        dest = stage / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(read.text, encoding='utf-8')
        twins.append(Unit(key=key, path=dest))
    return twins


def _cmd_bundle(args: argparse.Namespace) -> int:
    # No SDK extractor: the bundle is a pre-materialized directory of record files.
    units = bundle_units(args.bundle_dir)
    declaration_required = not recorded_regressions_only(args.bundle_dir, units)
    with tempfile.TemporaryDirectory(prefix='plurai-bundle-text-') as stage:
        twins = bundle_text_twins(units, Path(stage), err=sys.stderr)
        return _dispatch(args, units=[*units, *twins], kind='scenario-bundle', declaration_required=declaration_required)


def recorded_agent_versions(run_dir: str | Path | None) -> dict[str, str]:
    """``{agent_version: one scenario that recorded it}`` over the run's own transcripts.

    ``plurai-run-simulation`` stamps ``metadata.agent_version`` into every transcript it writes,
    so the run being uploaded already CONTAINS the answer the ``--agent-version`` flag asks the
    operator to retype. Read here rather than trusted from the flag, for the reason every other
    provenance field in this file is read rather than typed: a hand-passed value is a third copy
    of a fact, and the copies drift.

    Returns the DISTINCT values with one witness each — not a single value — because a run whose
    scenarios disagree has no agent version, and collapsing that to the first one seen would
    manufacture the certainty the disagreement denies.

    Silent on every unreadable transcript: this feeds a DEFAULT and a warning, never a refusal,
    and a run that lost a file still has a version to stamp from the others.
    """
    found: dict[str, str] = {}
    if run_dir is None:
        return found
    root = Path(run_dir)
    if not root.is_dir():
        return found
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            doc = json.loads((scenario / 'transcript.json').read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        metadata = doc.get('metadata') if isinstance(doc, dict) else None
        version = metadata.get('agent_version') if isinstance(metadata, dict) else None
        if isinstance(version, str) and version.strip():
            found.setdefault(version.strip(), scenario.name)
    return found


def _resolved_agent_version(args: argparse.Namespace) -> str | None:
    """``--agent-version`` reconciled against what the run's transcripts actually recorded.

    The flag is typically hand-typed free text, so it can name something other than the agent
    version (a model name, say) while the transcripts record the real one. The backend warns and
    stamps the declared value, but into a server log that reaches no operator:
    the finalize response has no warnings field.

    So the reconciliation happens HERE, where the operator is:

    * no flag, one recorded version -> stamp the recorded one. The archive then pins what
      answered, which is the whole point of the field.
    * no flag, several recorded -> stamp NOTHING and say which. A run whose scenarios disagree
      has no agent version, and picking one would be the manufactured certainty this file's other
      provenance rules exist to refuse.
    * a flag agreeing with the transcripts -> pass it through, silently.
    * a flag DISAGREEING -> the operator's word WINS, with a loud warning naming both. It is
      reported, not refused: an operator re-uploading an old run under a corrected tag is a real
      workflow, and this is the last field that should block an archive. The asymmetry with
      ``scenario_set_id`` (which refuses) is deliberate and stays on the backend side.
    """
    # `getattr`, because `_run_provenance` is also built from namespaces that carry no run dir
    # (identity checks, and any caller that only wants the version block). No run dir means no
    # transcripts to read, which is the same "nothing recorded" branch as a run that recorded none.
    recorded = recorded_agent_versions(getattr(args, 'run_dir', None))
    declared = args.agent_version
    if declared is None:
        if len(recorded) == 1:
            version = next(iter(recorded))
            print(
                f"ingest-run: --agent-version not given; stamping {version!r}, read from the run's own transcripts (metadata.agent_version).", file=sys.stderr
            )
            return version
        if len(recorded) > 1:
            named = ', '.join(f'{v!r} ({where})' for v, where in sorted(recorded.items()))
            print(
                f"ingest-run: WARNING — the run's transcripts record {len(recorded)} DIFFERENT agent versions "
                f'({named}), so this archive is stamped with NONE. Pass --agent-version to state which build the run '
                'exercised, or re-run the scenarios against one build.',
                file=sys.stderr,
            )
        return None
    if recorded and declared not in recorded:
        named = ', '.join(f'{v!r} ({where})' for v, where in sorted(recorded.items()))
        print(
            f"ingest-run: WARNING — --agent-version says {declared!r} but the run's transcripts record {named}. "
            'The flag WINS and is what the archive will claim; nothing else in it pins which build answered. '
            'Drop the flag to stamp what the transcripts recorded.',
            file=sys.stderr,
        )
    return declared


def _run_provenance(args: argparse.Namespace) -> RunProvenance:
    """The run's identity block from the parsed flags + the SDK's own version.

    ``sdk_version`` is computed here (from the installed package metadata, so it names the build
    that actually ran) and ``agent_version`` is reconciled against the run's own transcripts
    (:func:`_resolved_agent_version`); the rest are passed through verbatim, absent when the
    caller had nothing to say. Nothing is defaulted or GUESSED — see :class:`RunProvenance` for
    why a placeholder id would be worse than no id at all. Reading a field out of the artifact
    that records it is not a guess; it is the same move ``sdk_identity`` already makes.
    """
    return RunProvenance(
        sdk_version=__version__,
        # The identity that does not depend on dist metadata being fresh. Kept BESIDE
        # ``sdk_version`` rather than replacing it: backend readers key off the version string.
        sdk_identity=sdk_identity(),
        eval_set_id=args.eval_set_id,
        scenario_set_id=args.scenario_set_id,
        harness_fingerprint=args.harness_fingerprint,
        agent_version=_resolved_agent_version(args),
    )


def _stamp_trials_not_uploaded(root: Path, *, units: Sequence[Unit], staged_summary: Path) -> None:
    """Name, on the UPLOADED copy of ``run_summary.json``, the listed trials that packed nothing (``trials.not_uploaded``,
    ``[]`` when every one did): typed, so the backend reads an upload one trial short of ``trials.dirs`` as such, not
    from a stderr line. Written only where the run lists trials; the run's own file on disk is never rewritten."""
    listed = _listed_trial_dirs(root)
    if not listed or not staged_summary.is_file():
        return
    packed = {parts[1] for unit in units if len(parts := unit.key.split('/')) > 2}
    summary = json.loads(staged_summary.read_text(encoding='utf-8'))
    summary['trials']['not_uploaded'] = sorted(listed - packed)
    staged_summary.write_text(json.dumps(summary, ensure_ascii=False), encoding='utf-8')


def _warn_home_named(units: Sequence[Unit], workspace_root: str | Path, *, command: str) -> None:
    """Before anything leaves (no posture resolved, no archive, no lease yet): the paths naming the
    operator's home out of scope in the files about to upload, counted by the run's own detector
    (:meth:`plurai.host_effects.HostReadScope.home_named`, the agent's repo allowed — the root above
    the ``plurai_sim`` workspace, or the workspace a hand run names). A warning on stderr, never a stop."""
    root = Path(workspace_root).absolute()
    scope = host_read_scope(run_dir=None, repo=root.parent if root.name == PLUGIN_WORKDIR else root, harness=None)
    named: dict[str, int] = {}
    for unit in units:
        try:
            text = unit.path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        try:
            obj: Any = json.loads(text)
        except ValueError:
            obj = text
        n = scope.home_named(obj)
        if isinstance(obj, dict) and isinstance(obj.get('content_b64'), str):  # a document example's bytes ride base64
            with contextlib.suppress(binascii.Error, ValueError):
                n += scope.home_named(base64.b64decode(obj['content_b64'], validate=True).decode('utf-8', 'replace'))
        if n:
            named[unit.key] = n
    if named:
        listed = ', '.join(f'{key} ({n})' for key, n in sorted(named.items()))
        print(
            f'plurai {command}: WARNING (disclosure: home_named) — {sum(named.values())} path(s) naming your home folder outside the '
            f'scenario are in the files about to upload: {listed}. Nothing has left this machine yet; they upload as they are.',
            file=sys.stderr,
        )


def _scrub_home_paths(staged: Sequence[Unit]) -> None:
    """Our own run material (the staged copies, never the run on disk) with the operator's home folder written ``~``:
    only that prefix is replaced (``/Users/<name>/out/Q3.xlsx`` ships as ``~/out/Q3.xlsx``), every later segment and
    file name kept, so a backend re-judge reads the same names the local verdicts read. In every posture: the
    operator's home is no part of the material under test. ``run_summary.json`` too — its paths are no join key. A
    file that is not UTF-8 text is left as it is, and so is one that will not write back (said on stderr)."""
    home = str(Path.home()).rstrip('/\\')
    if len(home) < 2:
        return
    spellings = sorted({home, json.dumps(home)[1:-1]}, key=len, reverse=True)
    prefix = re.compile('|'.join(rf'(?<![\w/\\]){re.escape(h)}(?![\w.-])' for h in spellings))
    for unit in staged:
        try:
            text = unit.path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        scrubbed, n = prefix.subn('~', text)
        if not n:
            continue
        try:
            unit.path.write_text(scrubbed, encoding='utf-8')
        except (OSError, UnicodeEncodeError) as exc:
            print(f'plurai: {unit.key} uploads with its home paths unscrubbed ({type(exc).__name__})', file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    # No SDK extractor, and no anonymization in any posture: a run kind uploads the run as it is
    # (RUN_KINDS). $OUT is the customer's ONLY record of the finished run, and the staged copies are
    # what gets its trial stamp and home-path hygiene, so the original artifacts are never rewritten.
    units = run_units(args.run_dir)
    with tempfile.TemporaryDirectory(prefix='plurai-ingest-run-') as staging:
        staged: list[Unit] = []
        for unit in units:
            dest = Path(staging) / unit.key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(unit.path, dest)
            staged.append(Unit(key=unit.key, path=dest))
        _stamp_trials_not_uploaded(Path(args.run_dir), units=units, staged_summary=Path(staging) / RUN_SUMMARY_FILENAME)
        _scrub_home_paths(staged)
        return _dispatch(
            args,
            units=staged,
            kind='run-artifacts',
            declaration_required=False,
            provenance=_run_provenance(args),
        )


def _cmd_run_report(args: argparse.Namespace) -> int:
    # A separate archive from `ingest-run`, attached to that upload afterwards: the scoring archive
    # must stay exactly what every backend accepts, and the report is final only after the release
    # banner, which is written from the job that scoring upload feeds.
    units, absent_reason = run_report_units(args.run_dir)
    if absent_reason is not None:
        print(f'plurai ingest-run-report: no report to attach ({absent_reason}); nothing uploaded', file=sys.stderr)
        print(json.dumps({'absent_reason': absent_reason}))  # stdout = what attach_run_report takes instead of a manifest
        return 0
    # Copied first, like ingest-run: the home-path hygiene rewrites units in place, and these are the operator's report.
    with tempfile.TemporaryDirectory(prefix='plurai-ingest-run-report-') as staging:
        staged: list[Unit] = []
        for unit in units:
            dest = Path(staging) / unit.key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(unit.path, dest)
            staged.append(Unit(key=unit.key, path=dest))
        _scrub_home_paths(staged)
        return _dispatch(args, units=staged, kind='run-report', declaration_required=False)


def _cmd_folder_names(args: argparse.Namespace) -> int:
    # Local only: reads .plurai/scenario_folder_names.json, which never leaves the machine.
    name_map = load_name_map(args.workspace_root)
    if not args.text:
        print(json.dumps(name_map, indent=2, ensure_ascii=False))
        return 0
    for text in args.text:
        print(reveal(text, name_map))
    return 0


def _scenario_identity(conversion: ThreadConversion) -> dict[str, Any]:
    """The summary fields every convert-trace row carries, whether it wrote a bundle folder or a rendering."""
    return {
        'folder': conversion.folder,
        'source_file': conversion.provenance['source_file'],
        'thread_id': conversion.provenance['thread_id'],
        'message_count': len(conversion.messages),
        # Counted on the SOURCE, before any cut, and numbered as --cut-turn numbers them: the range
        # a caller picks its cut from, which is why a cut does not shrink it.
        'user_turns': conversion.source_user_turns,
        # The pipeline's next reader (the authoring side) lifts a trailing USER message out to
        # first_talk — reported so that normalization is visible here, not discovered later. The
        # converter deliberately does not do it: it carries verbatim.
        'final_message_type': conversion.messages[-1].type,
        # Beside the turn count because it says where that count starts: a resumed run's history
        # opens on this many messages the run did not record, and its turns are numbered after them.
        **({'resume_seed_messages': seeded} if (seeded := conversion.provenance.get('resume_seed_messages')) is not None else {}),
    }


def _cmd_render_only(args: argparse.Namespace) -> int:
    # Every flag here shapes a pin, and a rendering writes none. Refused rather than ignored, so a
    # caller never believes it cut or pinned something it did not — and before anything is read.
    pin_flags = [
        flag
        for flag, given in (
            ('--into', args.into is not None),
            ('--start-node', args.start_node is not None),
            ('--cut-turn', args.cut_turn is not None),
            ('--trim-mid-turn-tail', args.trim_mid_turn_tail),
        )
        if given
    ]
    if pin_flags:
        raise TraceConversionError(
            f'--render-only cannot be combined with {", ".join(pin_flags)}: a rendering writes no pin for it to shape. '
            'Render uncut to choose a turn, then convert without --render-only.'
        )
    result = _posture_gated(lambda: render_trace_export(args.export, args.out_dir, source_format=args.source_format, workspace_root=args.workspace_root))
    if isinstance(result, int):
        return result
    conversions = result.conversions
    summary = {
        'out_dir': str(Path(args.out_dir)),
        'source_format': args.source_format,
        'render_only': True,
        'scenario_count': len(conversions),
        'scenarios': [{**_scenario_identity(c), 'transcript': str(Path(args.out_dir) / c.folder / TRANSCRIPT_FILENAME)} for c in conversions],
        **_thread_findings(result),
    }
    if conversions:
        # No publish hint and no seed_state notice: nothing written here is a bundle, and a rendering
        # seeds nothing.
        print(
            f'plurai convert-trace: rendered {len(conversions)} transcript(s) to {args.out_dir}; nothing there is a bundle — '
            'convert with --cut-turn K, without --render-only, to write one.',
            file=sys.stderr,
        )
    print(json.dumps(summary, ensure_ascii=False))
    return _thread_exit(result, verb='rendered')


def _posture_gated(run: Callable[[], ConversionResult]) -> ConversionResult | int:
    """``run()``'s result, or — when the env's posture refuses it — the exit code the funnel returns for that.

    The posture refusal is the invocation's security finding: it is emitted to this invocation's
    collector, where an overrule naming it is refused, and the funnel stops the command on it.
    """
    try:
        return run()
    except PostureRefusedError as exc:
        current().emit(exc.finding)
        print(f'plurai convert-trace: error: {exc}', file=sys.stderr)
        return held_exit(refusal_code=exit_codes.REFUSED) or exit_codes.REFUSED


def _thread_findings(result: ConversionResult) -> dict[str, Any]:
    """The summary's per-thread gate fields, each finding also emitted to this invocation's collector.

    ``gate_findings`` carries every finding in the GateFinding wire shape; ``skipped`` names each thread
    that was not written and the gate that kept it out. One stderr line per skipped thread says why.
    """
    authority = current()
    for finding in result.findings:
        authority.emit(finding)
    for skip in result.skipped:
        print(f'plurai convert-trace: skipped {skip.source.name}: {skip.finding.remedy}', file=sys.stderr)
    return {
        'gate_findings': [finding.model_dump(mode='json') for finding in result.findings],
        'skipped': [{'source_file': skip.source.name, 'gate_id': skip.finding.gate_id} for skip in result.skipped],
    }


def _thread_exit(result: ConversionResult, *, verb: str) -> int:
    """0 when at least one thread was written; otherwise the funnel's refusal code.

    A thread's refusal holds that thread back, not the run: the funnel decides only when NOTHING was
    written, and then every finding it names is a skipped thread's.
    """
    if result.conversions:
        return 0
    print(f'plurai convert-trace: error: nothing was {verb} — every export was skipped (see the skipped lines above).', file=sys.stderr)
    return held_exit(refusal_code=exit_codes.REFUSED) or exit_codes.REFUSED


def _cmd_convert_trace(args: argparse.Namespace) -> int:
    # Local-only: converts on disk and STOPS. No scrub, no archive, no lease, no upload, no audit
    # entry — the resulting bundle dir is published by a separate `ingest-bundle` run, which is
    # where the egress machinery (and its posture pinning) lives. The posture is still read HERE,
    # because an anonymized env must not get a verbatim bundle written at all.
    if args.render_only:
        return _cmd_render_only(args)
    result = _posture_gated(
        lambda: convert_trace_export(
            args.export,
            args.out_dir,
            source_format=args.source_format,
            workspace_root=args.workspace_root,
            into_folder=args.into,
            start_node=args.start_node,
            trim_mid_turn_tail=args.trim_mid_turn_tail,
            cut_turn=args.cut_turn,
        )
    )
    if isinstance(result, int):
        return result
    conversions = result.conversions
    summary = {
        'out_dir': str(Path(args.out_dir)),
        # Echoed rather than assumed: a run through an adapter and a run through the built-in
        # reader produce identically-shaped bundles, so the summary is the only place the operator
        # can see WHICH reading of the bytes they are about to publish.
        'source_format': args.source_format,
        'scenario_count': len(conversions),
        'scenarios': [
            {
                **_scenario_identity(c),
                # Present only when the operator asked for the cut, and carrying the same figures
                # the pin's provenance carries — so a bundle whose history was shortened says so
                # in the summary an operator reads before publishing it, not only inside the file.
                **({'cut_turn': cut} if (cut := c.provenance.get('cut_turn')) is not None else {}),
                **({'trimmed_mid_turn_tail': trimmed} if (trimmed := c.provenance.get('trimmed_mid_turn_tail')) else {}),
                'agent_state_keys': sorted(c.agent_state),
                # Mirrors the pin's own key rather than echoing the flag: what an operator needs
                # to read back is what landed in the file the harness will seed from.
                'resume_point': c.resume_point.model_dump() if c.resume_point is not None else None,
                # Measured off the pin's own on-disk TEXT, so this is the size of the file that
                # is actually written rather than of a second rendering that could drift from it:
                # re-rendering with `json.dumps` here would miss the writer's `indent` and
                # trailing newline and under-report the size.
                'pin_bytes': len(c.pin_text.encode('utf-8')),
            }
            for c in conversions
        ],
        **_thread_findings(result),
    }
    if conversions:
        print(
            f'plurai convert-trace: wrote {len(conversions)} scenario folder(s) to {args.out_dir}; '
            f'publish with `plurai ingest-bundle --bundle-dir {args.out_dir}`',
            file=sys.stderr,
        )
        # Said HERE because nothing in the conversion can check it: whether the agent can be seeded is
        # a property of the customer's own module, which this machine never loads. The run refuses at
        # init rather than starting the agent empty and scoring the seeded history as its own work, so
        # the operator hears the requirement while the bundle is still local and cheap to hold.
        print(
            'plurai convert-trace: a resumed scenario is seeded by the harness — the agent module must export '
            '`seed_state(initial_agent_context, initial_world_state, resume_seed=None)` and apply the seed, or the run refuses at init.',
            file=sys.stderr,
        )
    # stdout = the local conversion summary (nothing egressed) — printed on every outcome, a run that
    # wrote nothing included, so a caller reads the skipped threads and their findings from one place.
    print(json.dumps(summary, ensure_ascii=False))
    return _thread_exit(result, verb='written')


def _add_plan_only(parser: argparse.ArgumentParser, *, expect: str) -> None:
    parser.add_argument(
        '--plan-only',
        action='store_true',
        help='Build the part exactly as a real run does (extraction, anonymization, scrub map and archive, so size and hash are the real ones) '
        'and print its entry with "planned": true, then stop. No request is made and no backend credentials are read; the audit log gets a '
        '"planned" entry, never an upload, and nothing is shown as an upload disclosure. Under `anonymized` posture the workspace key is minted '
        'if absent, so the real run scrubs under the same key.',
    )
    if expect == 'one':
        parser.add_argument(
            '--expect-sha256',
            type=sha256_argument,
            default=None,
            metavar='sha256:HEX',
            help='The hash the approved plan printed. A part that hashes differently is refused after it is built and before anything is disclosed '
            'or sent, with one stderr line naming both hashes.',
        )
    else:
        parser.add_argument(
            '--expect-sha256',
            type=named_sha256_argument,
            action='append',
            default=None,
            metavar='NAME=sha256:HEX',
            help='The hash the approved plan printed for the description part NAME. Repeat it once per part: the parts built must be exactly those '
            'named (an unexpected part, or a named part that is gone, is refused). A mismatch is refused after the parts are built and before '
            'anything is disclosed or sent, with one stderr line naming both hashes.',
        )


def _add_declared_names(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--declared-names',
        default=None,
        type=expand_user_path,
        help='A JSON list of the names the agent descriptions declare (tools, fields, arguments). Under `anonymized` posture a field '
        'whose name is on the list keeps its name through the scrub, so it is not renamed as a key; the scrub map lists '
        'the tool fields whose values were scrubbed.',
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    # The storage service is reached via the configured backend ($PLURAI_URL /
    # $PLURAI_TOKEN, else the saved login;
    # see StorageClient.from_env), not a per-command URL/key — so there is no storage flag.
    # Likewise the anonymization POSTURE is env-scoped config (top-level `anonymize:` in
    # plurai.config.yaml, resolved by plurai.ingest.posture), never a per-run flag.
    parser.add_argument(
        '--workspace-root',
        default='.',
        type=expand_user_path,
        help='Root the ingest state hangs off: .plurai/ (anonymization key + disclosure audit '
        'log) is created UNDER it, and plurai.config.yaml (anonymization posture) is looked up '
        'at <root>/plurai.config.yaml then <root>/plurai_sim/plurai.config.yaml. The plugin '
        'skills pass the plurai_sim directory; a hand run from the agent directory passes ".". '
        'Those two spellings do NOT share .plurai/ — under `anonymized` posture, switching '
        'spellings mints a second key and breaks pseudonym stability across runs, so pick one '
        'per workspace and keep it.',
    )


#: Behaviour a caller can require of this build, beyond the commands it lists (see :func:`_cmd_capabilities`).
FEATURES = ('parts-only-upload', 'content-addressed-parts', 'corpus-domain-member', 'plan-only-parts', 'export-dir-traces')


def _billing_url() -> str:
    """The billing page of the backend this process would talk to, or ``''``: capabilities must answer on a machine with no backend configured."""
    from plurai._endpoints import EndpointError, resolve  # noqa: PLC0415 - only this command needs the backend
    from plurai.login import _environments  # noqa: PLC0415

    try:
        return _environments.billing_url_for(resolve().url)
    except EndpointError:
        return ''


def _cmd_capabilities(commands: list[str], *, overrule_commands: list[str]) -> int:
    """Print the installed SDK's identity + the subcommands it actually has, as JSON.

    ``identity`` beside ``version``: the version is dist metadata and can name a build
    nobody is running, while ``commands`` is read off the live parser and is therefore the only
    thing here a caller should take a FLOOR decision on.

    ``features`` names behaviour that no command name reveals, because two builds list the same
    commands. ``parts-only-upload``: every upload (``ingest-*``, ``stash-doc``) sends its content
    as raw parts and never as one whole-object PUT, which is the only form a parts-only backend
    accepts. A caller that needs a feature checks for its name here; a build that predates the
    key has no ``features`` at all, and that reads as none.

    ``trace_selection_keys`` is read off the live trace-source registry for the same reason: an
    ``ingest-traces`` reader ignores a source-config key it does not know, so a caller that
    writes a filter for a build without it gets an unfiltered export and no error.

    ``overrule_commands`` — the subcommands taking ``--overrule`` — is read off the live parser too,
    so a caller feature-detects the flag instead of inferring it from a version.

    ``billing_url`` — the admin page where an organization adds credit, for the backend this process is configured for
    (``''`` when that environment has none or no backend is configured) — lets a caller word a credit refusal without
    keeping a table of its own.

    ``trace_formats`` — the built-in ``convert-trace --source-format`` readers — is read off the live
    registry, so a caller dispatching an ingested ``source_kind`` learns whether this build reads it.
    """
    payload = {
        'version': __version__,
        'identity': sdk_identity(),
        'commands': commands,
        'features': list(FEATURES),
        'overrule_commands': overrule_commands,
        'trace_selection_keys': trace_selection_keys(),
        'trace_formats': sorted(TRACE_FORMAT_REGISTRY),
        'billing_url': _billing_url(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _command_names(sub: argparse._SubParsersAction) -> list[str]:
    """Every command the parser has, a grouped verb as ``'<group> <verb>'`` beside its group: a
    floor check on the group alone passes a build that has only some of its verbs."""
    names: list[str] = []
    for name, parser in sub.choices.items():
        names.append(name)
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                names.extend(f'{name} {verb}' for verb in action.choices)
    return sorted(names)


def build_parser() -> argparse.ArgumentParser:
    parser = exit_codes.UsageParser(prog='plurai', description='Plurai SDK — anonymized object-store ingestion.')
    # Introspection BEFORE the required-subcommand check: `plurai --version` must answer
    # on its own, because the question it answers ("which SDK is this venv running?") is
    # the first one asked when a command or a transcript field is missing, and argparse
    # would otherwise reject the bare invocation. argparse's `version` action exits during
    # argument consumption, so the subparser requirement never gets to fire.
    parser.add_argument('--version', action='version', version=f'plurai {__version__}')
    # The same question answered WITHOUT the dist-metadata caveat, and on its own flag so a
    # caller that only wants the identity does not have to parse the capabilities JSON.
    # ``--version`` keeps its exact output: it is a stable contract the plugin's resolver reads.
    parser.add_argument(
        '--identity',
        action='version',
        version=sdk_identity_line(),
        help='Print WHICH plurai is running (commit or content hash), not just its declared version.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # The machine-readable half of the same answer: a version alone does not say whether
    # THIS build has a given subcommand, and the surface is read off the live parser, so
    # it can never drift from what is registered below.
    capabilities = sub.add_parser('capabilities', help="Print this SDK's version + available subcommands as JSON.")
    capabilities.set_defaults(
        func=lambda _args: _cmd_capabilities(_command_names(sub), overrule_commands=sorted(name for name, p in sub.choices.items() if takes_overrule(p)))
    )

    login_parser = sub.add_parser('login', help='Sign in through the browser and save a backend key where every Plurai client reads it.')
    login_commands.add_arguments(login_parser)
    login_parser.set_defaults(func=login_commands.run)

    # Listed for --help only: `_main` hands `logout` to login_commands.main_logout before this parser runs.
    sub.add_parser('logout', help='Revoke the key plurai login saved and remove the credentials file.')

    diagnose_parser = sub.add_parser(
        'diagnose',
        help='Print a bundle of versions, ids and presence flags to paste into a support issue.',
        description='Print the bundle to paste into a support issue: versions, presence flags and the ids support can look up.',
        epilog=diagnose_commands.EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    diagnose_commands.add_arguments(diagnose_parser)
    diagnose_parser.set_defaults(func=diagnose_commands.run)

    # No --set-id / --job-id on any subcommand: the server mints the job/set id from the manifest.
    # The client only extracts, anonymizes, bundles, and uploads.
    corpus = sub.add_parser('ingest-corpus', help='Extract, anonymize, and publish a RAG/SQL corpus.')
    _add_common(corpus)
    corpus.add_argument('--source-kind', required=True)
    corpus.add_argument('--source-config', default=None, type=expand_user_path, help='Path to a JSON file of extractor source config (credentials stay local).')
    corpus.add_argument('--out-dir', required=True, type=expand_user_path)
    corpus.add_argument(
        '--root',
        action='append',
        default=[],
        type=expand_user_path,
        help='A directory source root: a directory (its tree) or one file. Repeatable; every root lands in ONE corpus. In place of source_config.path.',
    )
    # The SCOPE: which suffixes of a directory source's roots are ingested. Required for a directory
    # source, and taken from the scan, so the corpus and the scan contract describe the same files.
    corpus.add_argument(
        '--scan-contract',
        default=None,
        type=expand_user_path,
        help="Directory source: the scan's scan_contract.json; its recorded source_inventory.suffixes is the scope ingested.",
    )
    corpus.add_argument(
        '--suffixes',
        nargs='*',
        action='extend',
        default=None,
        help='Directory source, in place of --scan-contract: the EXTRA suffixes the scan was run with (its own --suffixes, e.g. '
        '`--suffixes html csv`), on top of its default set; `--suffixes` alone is the default set. A suffix ingest cannot read is refused.',
    )
    corpus.add_argument(
        '--domain',
        default=None,
        help='Inline JSON of domain hints recorded in the local manifest (e.g. \'{"task_description": "...", "semantic_enrichment": false}\'). No credentials.',
    )
    corpus.add_argument(
        '--domain-file',
        default=None,
        type=expand_user_path,
        help=f'A JSON object with an optional "domain" object and an optional "questions" list of strings (at most {MAX_DOMAIN_BYTES} bytes), '
        f'delivered unscrubbed as {DOMAIN_MEMBER} in the corpus part: it is derived from the agent descriptions, which also travel unscrubbed.',
    )
    # Chunking knobs. Omit them to get the extractor's own defaults — for a doc folder that
    # means SECTION chunking (one chunk per markdown heading) plus structural relationships,
    # which is what makes the corpus graph walkable; one-chunk-per-file is not.
    corpus.add_argument(
        '--chunk-mode',
        default=None,
        choices=['section', 'tokens'],
        help="Text chunking: 'section' (markdown headings, the default) or 'tokens' (flat sliding window).",
    )
    corpus.add_argument('--chunk-size', type=int, default=None, help='Max tokens per chunk; a heading section larger than this falls back to token windows.')
    corpus.add_argument('--chunk-overlap', type=int, default=None, help='Token overlap between windows of an oversize section (or of non-markdown text).')
    # Chunking is the knob that decides whether the corpus graph is walkable at all, and the shape is
    # only visible after extracting — so extraction is separable from egress: check, re-chunk, publish.
    corpus.add_argument(
        '--extract-only',
        action='store_true',
        help='Extract into --out-dir and STOP before any egress (no scrub, no archive, no lease, no upload, no audit entry, '
        'so --workspace-root is used only for a gate record, written under its .plurai/ when this invocation carries an '
        '--overrule or a gate reports a finding). stdout carries the local extraction summary instead of the manifest — chunk, '
        'relationship and source-document counts plus relationships-per-chunk — so an under-chunked corpus is VISIBLE '
        'before it is uploaded and built. It REPORTS those numbers, it does not judge them: this exits 0 for a healthy '
        'and an under-chunked corpus alike (only a corpus with no usable chunk at all fails). The verdict stays with the '
        'backend corpus-health gate; re-chunk and re-extract until the numbers look right, then publish.',
    )
    corpus.add_argument(
        '--label', default='kb', help="The corpus part's label in the grounding package; the same bytes under a new label are the same part under that label."
    )
    _add_declared_names(corpus)
    _add_plan_only(corpus, expect='one')
    corpus.set_defaults(func=_cmd_corpus)

    traces = sub.add_parser('ingest-traces', help='Export, anonymize, and publish production traces.')
    _add_common(traces)
    traces.add_argument('--source-kind', required=True)
    traces.add_argument('--source-config', default=None, type=expand_user_path, help='Path to a JSON file of sink source config (credentials stay local).')
    traces.add_argument('--out-dir', required=True, type=expand_user_path)
    traces.add_argument('--limit', type=int, default=None)
    _add_declared_names(traces)
    _add_plan_only(traces, expect='one')
    traces.set_defaults(func=_cmd_traces)

    descriptions = sub.add_parser('ingest-descriptions', help='Deliver the agent descriptions as content-addressed parts.')
    _add_common(descriptions)
    descriptions.add_argument(
        '--descriptions-dir',
        default=None,
        type=expand_user_path,
        help='The folder holding plurai.config.yaml, simulation_prd.md and scenario_variables.yaml (those present are delivered). Default: --workspace-root.',
    )
    descriptions.add_argument(
        '--scan-contract',
        default=None,
        type=expand_user_path,
        help=f"The scan's scan_contract.json (it lives under the workspace's artifacts folder), delivered as {SCAN_CONTRACT_NAME}.",
    )
    _add_plan_only(descriptions, expect='many')
    descriptions.set_defaults(func=_cmd_descriptions)

    docs = sub.add_parser('ingest-doc-examples', help='Scan, anonymize, and publish document-example families.')
    _add_common(docs)
    docs.add_argument('--source-dir', required=True, type=expand_user_path)
    docs.add_argument('--out-dir', required=True, type=expand_user_path)
    docs.set_defaults(func=_cmd_doc_examples)

    # The one LOCAL verb: it converts on disk and stops, so it egresses nothing and the bundle it
    # writes is published by a separate `ingest-bundle` run. It still takes --workspace-root,
    # because the posture it reads there decides whether the bundle may be written at all.
    convert = sub.add_parser(
        'convert-trace',
        help='Convert a trace export into a resume-pinned scenario bundle directory (local; publish with ingest-bundle).',
    )
    _add_common(convert)
    convert.add_argument(
        '--export',
        required=True,
        type=expand_user_path,
        help='A trace export in --source-format (for langsmith: one JSON file holding the list of runs in that thread), or a directory '
        'of them — e.g. the --out-dir an `ingest-traces` run wrote. A directory converts every artifact in it (manifest.json aside), '
        'one folder each, and every artifact in it must be the same format. Each thread is converted or SKIPPED on its own: one that '
        'cannot be carried faithfully is named on stderr and in the stdout summary (`skipped`, `gate_findings`) while the others are '
        'written. The exit code is 65 (refused) only when nothing was written.',
    )
    convert.add_argument(
        '--source-format',
        default=DEFAULT_SOURCE_FORMAT,
        # Built from the registry rather than written out: an entry added there and not here would
        # leave the help naming fewer readers than the CLI has.
        help=f'How to read the export. Built-in readers: {", ".join(sorted(TRACE_FORMAT_REGISTRY))}; defaults to {DEFAULT_SOURCE_FORMAT!r}. '
        f'Any OTHER name is an ADAPTER: a YAML mapping document committed to the workspace at '
        f'<workspace-root>/{ADAPTER_DIR_NAME}/<name>.yaml (or under plurai_sim/), which says where in that format the conversation '
        'lives. The mapping is executed by the SDK, so the conversion is deterministic and the message bytes are carried verbatim. '
        'An unknown name fails closed and lists what this workspace can read.',
    )
    convert.add_argument(
        '--out-dir',
        required=True,
        type=expand_user_path,
        help='Bundle directory to write: one <thread-…>/ folder per export, each holding ROLES.md, resume_pin.json and '
        'transcript.md, plus a set-level BUNDLE.md. Hand this same path to `ingest-bundle --bundle-dir` to publish it.',
    )
    convert.add_argument(
        '--into',
        default=None,
        help='Convert into an EXISTING hand-authored folder under --out-dir instead of writing a new one, so that folder '
        'carries the pin beside its own materials and its scenario resumes the conversation while staying grounded in them. '
        "Names one folder, so it requires a single --export. The folder's ROLES.md is APPENDED to — the hand-authored "
        'description stays first and is never overwritten. Refuses a folder that does not exist (author it first) and a '
        'folder already holding a different conversation; re-running the same --export with the same --into, --start-node '
        'and --cut-turn changes nothing. A different --start-node or --cut-turn is a different pin, which that same guard '
        'refuses rather than overwrites.',
    )
    convert.add_argument(
        '--start-node',
        default=None,
        help='Top-level graph node the harness enters on turn 1 of the resumed leg, pinned into every thread this '
        'invocation converts. Carried verbatim: the SDK checks only that the name is non-blank, because the graph is '
        "known to the customer's harness alone and that is where the name is resolved at seed time. Nothing is derived "
        'from the trace — a node the traced conversation never reached is a legitimate target. Omit it and the pin '
        'carries `resume_point: null`, which leaves the harness to start the leg wherever it ordinarily would.',
    )
    convert.add_argument(
        '--cut-turn',
        type=int,
        default=None,
        metavar='K',
        help='Resume the conversation from turn K instead of from its end: turns 0..K-1 are seeded with their agent '
        "replies, and what the user said on turn K becomes the scenario's opening message, said live. Turns are counted "
        'from 0 over the USER messages, so --cut-turn 1 replays everything after the first exchange; every transcript.md '
        'heads each user message with that number (`### human · turn K`), and --render-only writes one without converting. '
        'Applies to every '
        'export this invocation converts, and is recorded in the pin provenance. The pinned agent state follows the cut '
        'only for a source that records one state per turn (a run transcript); a source that records a single final '
        'state reached it AFTER the cut, so its pin carries no state at all — provenance says which, under '
        '`agent_state_at`, and names the channels left out under `state_withheld`. For a run that was itself resumed, '
        'turns count from the first live turn — the numbering its '
        "judges report — and the run's resume seed is kept whole ahead of them, so --cut-turn 0 cuts at the seed and "
        "pins the seed's state (a seed of state alone has no history to cut at, and is refused). Refused for turn 0 of "
        'anything else (which would seed nothing) and for a turn past the end of the conversation — that thread is '
        'skipped, never written. From a production source (any format but plurai-run), ROLES.md then ends on a section headed '
        '"World the live leg needs": the tool calls and results the source recorded after the cut, verbatim and '
        'capped, as constraints for the world author. The pin is the same with or without it.',
    )
    convert.add_argument(
        '--trim-mid-turn-tail',
        action='store_true',
        help='Convert a trace that was captured MID-TURN — between an agent tool call and the tool reply — by dropping the '
        'incomplete tail, so the pin resumes from the completed agent turn before it. Without this flag such an export is '
        'refused, because the backend refuses the pin during generation and would do so only after the upload. The cut is '
        'never implicit: it is recorded in the pin provenance and reported in this summary, and a trailing user message is '
        'kept (it opens the resumed conversation).',
    )
    convert.add_argument(
        '--render-only',
        action='store_true',
        help='Write ONLY <out-dir>/<folder>/transcript.md for each export — the conversation rendered with every user '
        'turn numbered exactly as --cut-turn counts them — to read before choosing a cut. Same posture gate, reader and '
        'projection as a conversion, but no pin, ROLES.md or BUNDLE.md is written, so the out-dir is not a bundle. A '
        'capture that ends mid-turn is rendered rather than refused, since a rendering is only read. Refuses --into, '
        '--start-node, --cut-turn and --trim-mid-turn-tail (each shapes a pin), and an --out-dir that already holds a '
        'BUNDLE.md. The stdout summary carries render_only: true, and per export its transcript path and user_turns.',
    )
    convert.set_defaults(func=_cmd_convert_trace)

    bundle = sub.add_parser('ingest-bundle', help='Anonymize and publish a pre-materialized scenario bundle.')
    _add_common(bundle)
    bundle.add_argument('--bundle-dir', required=True, type=expand_user_path)
    bundle.set_defaults(func=_cmd_bundle)

    run = sub.add_parser('ingest-run', help="Publish a finished run's saved artifacts (score_run by-reference: upload once, re-score per scenario).")
    _add_common(run)
    run.add_argument(
        '--run-dir',
        required=True,
        type=expand_user_path,
        help="The runner's output dir ($OUT): one folder per scenario, each holding transcript.json + saved artifacts.",
    )
    # Run PROVENANCE — every one of them OPTIONAL, and absent means absent: an unstamped run
    # uploads fine (the backend warns), while a fabricated id would read as a real one that
    # disagrees. The SDK's own version rides automatically; there is no flag for it, and none for
    # the backend's ground-truth revision or build id — no client can know either authoritatively,
    # so the server enriches those at finalize (see plurai.ingest.manifest_models.RunProvenance).
    run.add_argument(
        '--eval-set-id',
        default=None,
        help='The eval set this run was scored against (plurai_sim/.eval_set_id). Optional; omit it rather than guessing.',
    )
    run.add_argument(
        '--scenario-set-id',
        default=None,
        help='The scenario set this run was driven from (plurai_sim/.scenario_set_id). Optional; omit it rather than guessing.',
    )
    run.add_argument(
        '--harness-fingerprint',
        default=None,
        help='Opaque digest of the harness (simulate.py + its mock files) the driver computed for this run, so a later '
        'replay can tell the harness has since changed. Passed through verbatim — the SDK neither computes nor parses it.',
    )
    run.add_argument(
        '--agent-version',
        default=None,
        help='The agent build this run exercised (the same tag `plurai-run-simulation --agent-version` stamped). Optional, '
        "and normally OMITTED: the run's own transcripts record it (`metadata.agent_version`) and that is what is stamped "
        'when this is absent. Pass it only to override, and a value disagreeing with the transcripts is WARNED about and '
        'used. Still never defaulted to `git describe` or the literal unknown — provenance must distinguish absent from '
        'present-and-wrong.',
    )
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser(
        'ingest-run-report',
        help="Publish a run's rendered report (the files its report_files.json lists), to attach to that run's upload.",
    )
    _add_common(report)
    report.add_argument(
        '--run-dir',
        required=True,
        type=expand_user_path,
        help="The runner's output dir ($OUT) the report was rendered into. With no report there, stdout is "
        '{"absent_reason": "stale_run" | "not_rendered"} and nothing is uploaded.',
    )
    report.set_defaults(func=_cmd_run_report)

    # Not an ingest verb: it writes to the customer's OWN observability platform, never to the backend.
    correct = sub.add_parser(
        'correct-scores',
        help="Overwrite corrected evals' scores on the connected observability platform (Langfuse / LangSmith).",
        description="Every eval whose state in a conversation's verdicts.json differs from the run's own (history[0]) gets the "
        'score its current cell maps to, written over the score the run posted. With no platform connected it prints a skip '
        'line and exits 0. Prints one line per eval: written | not_recorded | unsupported | failed. Exit codes: 0 always, '
        "except 66 when the run dir or a conversation's verdicts.json cannot be read.",
    )
    correct.add_argument(
        '--run-dir',
        required=True,
        metavar='DIR',
        type=expand_user_path,
        help="The runner's output dir ($OUT): every conversation attempt's verdicts.json under it (listed trials and stability retries included).",
    )
    correct.set_defaults(func=_cmd_correct_scores)

    names = sub.add_parser(
        'folder-names',
        help='Read the local map of pseudonymized scenario folder names (anonymized bundle uploads). Local only; uploads nothing.',
    )
    _add_common(names)
    names.add_argument(
        'text',
        nargs='*',
        help='Ids or text to print with every known pseudonym replaced by its folder name (e.g. a run folder name). With none, print the whole map as JSON.',
    )
    names.set_defaults(func=_cmd_folder_names)

    # Not an ingest verb: a generation input is not customer data, so it gets no posture,
    # disclosure or lease. A usage error exits 64 in every command.
    stash = sub.add_parser(
        'stash-doc',
        help='Stash ONE generation input (a PRD, a scenario_variables.yaml) on the backend from disk; print its input doc id.',
        description='Upload the file as a storage object and name it to POST /input-docs; print the content-addressed input doc id '
        'on stdout, alone. Everything else (byte count, backend warnings, errors) goes to stderr. Exit codes: 0 stashed; '
        '66 file missing/empty/unreadable; 69 transport failure or no id in the answer; 12 the backend predates this stash '
        '(use the MCP stash_input_doc tool); 77 bearer rejected; 17 the document was refused (fix the file); 64 usage error.',
    )
    stash.add_argument('file', type=expand_user_path, help='The document to stash (UTF-8 text). Pass an absolute path.')
    stash.add_argument('label', help='The discovery key the backend files it under, e.g. simulation_prd. Characters: [A-Za-z0-9._-].')
    stash.add_argument(
        '--out',
        default=None,
        type=expand_user_path,
        metavar='ID_FILE',
        help='Also write the id to ID_FILE, atomically and only on success. A failed run REMOVES any file already '
        'there, so a stale id cannot pass for a fresh one. Prefer this over capturing stdout.',
    )
    stash.set_defaults(func=_cmd_stash_doc)

    _add_job_and_backend(sub)
    _add_fetch(sub)
    # The ONE writer of a conversation attempt's verdicts.json, for the plugin's rescore / correction / converter.
    verdicts.add_parser(sub, path_type=expand_user_path)
    # THE scores block (run_summary.json → scores), recomputed from those files after every verdicts write.
    scoring.add_parser(sub, path_type=expand_user_path)

    # One spelling of the flag on every subcommand that converts or publishes input; the two that
    # only read (capabilities, folder-names) gate nothing.
    for name in INGEST_OVERRULE_SUBCOMMANDS:
        add_overrule_argument(sub.choices[name])

    return parser


def _add_job_and_backend(sub: argparse._SubParsersAction) -> None:
    """``job status|wait`` and ``backend identity|probe``: the exit tables live in the docstrings of
    :mod:`plurai.ingest.job_commands` and :mod:`plurai.ingest.backend_commands`."""
    job = sub.add_parser('job', help='Read one backend job, or block until it is terminal.')
    job_verbs = job.add_subparsers(dest='verb', required=True)
    status = job_verbs.add_parser('status', help='Print the job envelope of GET /jobs/{id} as JSON.')
    status.add_argument('job_id')
    status.set_defaults(func=lambda args: job_commands.run_job_status(args.job_id))
    wait = job_verbs.add_parser(
        'wait',
        help='Block until the job is done or failed, printing a timestamped line per status change. Never pipe it: the exit code is the result.',
    )
    wait.add_argument('job_id')
    wait.add_argument('--max-seconds', type=int, default=540, metavar='N', help='Wall-clock budget, clamped to 30..540 (default 540).')
    wait.set_defaults(func=lambda args: job_commands.run_job_wait(args.job_id, args.max_seconds))

    backend = sub.add_parser('backend', help='Ask the backend which deployment it is, or whether it answers.')
    backend_verbs = backend.add_subparsers(dest='verb', required=True)
    identity = backend_verbs.add_parser('identity', help='Print GET /backend-identity as one line of JSON.')
    identity.set_defaults(func=lambda _args: backend_commands.run_backend_identity())
    probe = backend_verbs.add_parser('probe', help='Print the HTTP code of GET PATH (000 = no response); exit 0 only on a 2xx on that path.')
    probe.add_argument('--path', default='/healthz', help='Path to probe (default /healthz).')
    probe.add_argument('--timeout', type=float, default=5.0, metavar='S', help='Seconds for the whole exchange, redirects included (default 5).')
    probe.set_defaults(func=lambda args: backend_commands.run_backend_probe(args.path, args.timeout))


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    """``fetch policies|artifact``: the exit tables live in the docstring of :mod:`plurai.ingest.fetch_commands`."""
    fetch = sub.add_parser('fetch', help='Download backend-owned files to disk.')
    verbs = fetch.add_subparsers(dest='verb', required=True)
    policies = verbs.add_parser(
        'policies',
        help='Sync a local policies_config.yaml from GET /eval-sets/{id}/policies.yaml; print state=<promoted|draft|unknown|absent>. '
        'Refuses a DRAFT (exit 16) unless --allow-draft.',
    )
    policies.add_argument('eval_set_id')
    policies.add_argument('--out', required=True, type=expand_user_path, metavar='FILE', help='The file to write (a temp file beside it, then a rename).')
    policies.add_argument('--allow-draft', action='store_true', help="Write an open edit session's draft too (for a scratch copy).")
    policies.set_defaults(func=lambda args: fetch_commands.run_fetch_policies(args.eval_set_id, args.out, args.allow_draft))
    artifact = verbs.add_parser(
        'artifact', help='Download files through GET /artifacts/{id}/files/{relpath}; print "wrote <path>" or "missing <relpath>" per file.'
    )
    artifact.add_argument('artifact_id')
    artifact.add_argument('relpaths', nargs='+', metavar='relpath')
    where = artifact.add_mutually_exclusive_group(required=True)
    where.add_argument('--dir', default=None, type=expand_user_path, metavar='DIR', help='Write each file to DIR/<relpath>.')
    where.add_argument('--out', default=None, type=expand_user_path, metavar='FILE', help='Write the one file to FILE.')
    artifact.set_defaults(func=lambda args: fetch_commands.run_fetch_artifact(args.artifact_id, args.relpaths, args.dir, args.out))


def _verdict_dirs(root: Path) -> list[Path]:
    """Every conversation attempt folder of a run: each scenario folder (one holding a transcript or an error
    record), in ``root`` and in each trial directory its summary lists, and each one's ``stability_retry/``."""
    dirs: list[Path] = []
    trial_dirs = _listed_trial_dirs(root)
    for base in (root, *(root / name for name in sorted(trial_dirs))):
        if not _safe_is_dir(base):
            continue
        for folder in sorted(p for p in base.iterdir() if _safe_is_dir(p) and p.name not in trial_dirs):
            if _safe_is_file(folder / 'transcript.json') or _safe_is_file(folder / 'error.json'):
                dirs.append(folder)
            if _safe_is_dir(retry := folder / RETRY_FOLDER) and any(_safe_is_file(retry / name) for name in RUN_ARTIFACT_FILES):
                dirs.append(retry)
    return dirs


def _cmd_correct_scores(args: argparse.Namespace) -> int:
    """Overwrite the observability scores of corrected eval cells, read off the run's verdicts.json
    files. Never fails the caller: a platform error is printed as a ``failed`` line, and only an unreadable
    run exits non-zero."""
    from pydantic import ValidationError

    from plurai.observability.factory import create_observability_sink
    from plurai.observability.score_correction import overwrite_scores, overwrites_from_verdicts
    from plurai.observability.sink_protocol import NoOpSink
    from plurai.settings import Settings

    root = Path(args.run_dir)
    if not root.is_dir():
        print(f'plurai correct-scores: error: {root} is not a run directory', file=sys.stderr)
        return exit_codes.INPUT_UNREADABLE
    try:
        items = overwrites_from_verdicts(verdicts.read(item_dir) for item_dir in _verdict_dirs(root))
    except verdicts.VerdictsError as e:
        print(f'plurai correct-scores: error: {e}', file=sys.stderr)
        return exit_codes.INPUT_UNREADABLE
    if not items:
        print('no eval changed since the run — nothing to overwrite')
        return 0
    try:
        settings = Settings()
    except ValidationError as e:
        print(f'observability settings are invalid ({"; ".join(err["msg"] for err in e.errors())}) — skipped')
        return 0
    sink = create_observability_sink(settings)
    if isinstance(sink, NoOpSink):
        print('no observability platform connected — skipped')
        return 0
    for outcome in overwrite_scores(items, sink):
        detail = f': {outcome.detail}' if outcome.detail else ''
        print(f'{outcome.status} {outcome.thread_id} {outcome.name}{detail}')
    return 0


def _cmd_stash_doc(args: argparse.Namespace) -> int:
    from plurai.ingest.stash_doc import run_stash_doc

    return run_stash_doc(args.file, args.label, args.out)


def _prepare_gate_record_dir(record_dir: Path) -> None:
    """Make ``.plurai/`` private and self-ignoring before the first record lands in it, as the audit log does."""
    key_dir = record_dir.parent
    key_dir.mkdir(parents=True, exist_ok=True)
    _ensure_key_dir_gitignored(key_dir)
    record_dir.mkdir(exist_ok=True)
    with contextlib.suppress(OSError):
        record_dir.chmod(0o700)
    if is_windows():  # no modes there; a failure is said, as the audit log's folder says it, never swallowed
        try:
            _private_files.restrict_directory(record_dir)
        except OSError as error:
            print(f'plurai: could not limit {record_dir} to your account ({error}); the gate records may be readable by others', file=sys.stderr)


def _prepare_refusal_dir(key_dir: Path) -> None:
    """Make ``.plurai/`` exist, private and self-ignoring, before a refusal record is appended to it."""
    key_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        key_dir.chmod(0o700)
    _ensure_key_dir_gitignored(key_dir)


def _refusal_sink(args: argparse.Namespace) -> network_refusal.RefusalSink | None:
    """Where this command records a request that did not reach Plurai: ``<workspace root>/.plurai/``, when it takes one."""
    root = getattr(args, 'workspace_root', None)
    if root is None:
        return None
    return network_refusal.RefusalSink(path=Path(root) / KEY_DIR_NAME / network_refusal.RECORD_FILENAME, command=args.command, prepare=_prepare_refusal_dir)


def _run_command(args: argparse.Namespace) -> int:
    token = network_refusal.SINK.set(_refusal_sink(args))
    try:
        return _run_classified(args)
    finally:
        network_refusal.SINK.reset(token)


def _run_classified(args: argparse.Namespace) -> int:
    try:
        return int(args.func(args))
    except StorageConfigError as e:
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.CREDENTIALS
    except StorageCreditStopError as e:
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.JOB_FAILED
    except StorageUploadError as e:
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.for_http_status(e.status_code) or exit_codes.BACKEND_REFUSED
    except PartUploadUnsupportedError as e:
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.ROUTE_MISSING
    except _FAIL_CLOSED_ERRORS as e:
        # Present a clean, actionable one-liner instead of a traceback — the message
        # itself carries the fix (which extra/driver to install, what to declare, …).
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.REFUSED
    except httpx.HTTPError as e:
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == client_handshake.UPGRADE_STATUS:
            print(f'plurai {args.command}: error: {client_handshake.upgrade_message_for(e.response)}', file=sys.stderr)
            return exit_codes.UPGRADE_REQUIRED
        if isinstance(e, NetworkRefused | StorageContentUndelivered):
            # Not Plurai's answer, or an upload a network control held: the message names the host and who can fix it.
            print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        else:
            # A transport failure, or a 5xx that outlasted the retries. An upload that stops here has
            # left at most an object without content, which nothing reads, so a fresh run is the fix;
            # a traceback would hide that.
            try:
                target = f' {e.request.method} {e.request.url}'
            except RuntimeError:  # an HTTPError raised without a request
                target = ''
            print(f'plurai {args.command}: error: request{target} failed: {type(e).__name__}: {e}; run it again', file=sys.stderr)
        if isinstance(e, httpx.HTTPStatusError) and (row := exit_codes.for_http_status(e.response.status_code)) is not None:
            return row
        return exit_codes.TRANSPORT


def main(argv: list[str] | None = None) -> int:
    # Under the redaction, so the traceback of an internal fault and an echoed error body never print a credential.
    with redacted_process_output():
        return exit_codes.guarded('plurai', lambda: _main(argv))


def _main(argv: list[str] | None) -> int:
    utf8_stdio()
    tokens = sys.argv[1:] if argv is None else argv
    # Before the shared parser: its usage error repeats the arguments, and a token given as one is already too exposed.
    if tokens[:1] == ['login']:
        return login_commands.main(tokens[1:])
    if tokens[:1] == ['logout']:
        return login_commands.main_logout(tokens[1:])
    parser = build_parser()
    args = parser.parse_args(tokens)
    if not hasattr(args, 'overrule'):
        return _run_command(args)
    # Read before the command runs: a malformed overrule file refuses it rather than reading as
    # "nothing was overruled", which is a different instruction from the one the caller wrote.
    try:
        overrules_file = load_overrules(args.overrule) if args.overrule is not None else None
    except OverruleFileError as e:
        print(f'plurai {args.command}: error: {e}', file=sys.stderr)
        return exit_codes.REFUSED
    record = RecordTarget(
        path=Path(args.workspace_root) / KEY_DIR_NAME / GATE_RECORD_DIRNAME / f'{args.command}.json',
        when_empty=False,
        prepare=_prepare_gate_record_dir,
    )
    # The fail-closed mapping runs INSIDE the invocation, so the record carries the code the process exits with.
    return run_invocation(command=f'plurai {args.command}', overrules_file=overrules_file, record=record, body=lambda: _run_command(args))


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    'FEATURES',
    'DocExampleBinaryError',
    'RECORDED_REGRESSION_FILES',
    'Unit',
    'build_parser',
    'bundle_units',
    'corpus_extraction_summary',
    'corpus_shape_note',
    'corpus_units',
    'doc_example_units',
    'main',
    'run_ingest',
    'run_report_units',
    'run_units',
    'pseudonymize_trace_units',
    'recorded_regressions_only',
    'trace_units',
]
