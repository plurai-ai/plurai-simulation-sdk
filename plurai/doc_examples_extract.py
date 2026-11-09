"""Client-side document-example extraction.

Runs the LOCAL half of the document-example-family upload pipeline — the only
part that needs the customer's files. It scans a directory of example documents
organized into family folders, base64-encodes each file, and writes the upload
payload + a summary manifest that ``/upload-doc-examples`` (via the plugin) then
chunks and sends to the Plurai backend's
``upload_doc_examples_append_files(files_json=...)`` MCP tool.

No LLM keys or network access are required here — extraction is pure local I/O.

**Expected input layout**::

    <source_dir>/<family>/FAMILY.md        # REQUIRED per family — describes it
    <source_dir>/<family>/<example files>  # the example documents (any type)

Each immediate subdirectory of ``source_dir`` is one **family**
(e.g. ``invoice/``, ``boarding_pass/``). Files directly under ``source_dir``
(not inside a family folder) are ignored with a warning.

The wire artifacts written under ``out_dir`` are:

* ``families.json`` — ``[{"family", "path", "content_b64"}, …]`` — the JSON
  array the plugin reads, chunks into batches, and sends to
  ``upload_doc_examples_append_files``.
* ``manifest.json`` — ``{"schema_version", "family_count", "families": [{"name",
  "example_count", "file_count"}], "total_bytes", "files": {"families": "families.json"}}``.

Size caps (matching the backend):

* Single file  > 8 MB  → skipped with a warning.
* Total payload > 128 MB → ``ValueError`` raised before any file is written.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

FAMILIES_FILENAME = 'families.json'
MANIFEST_FILENAME = 'manifest.json'
FAMILY_DESCRIPTION_FILENAME = 'FAMILY.md'

# Per-file size cap: files larger than this are skipped with a warning.
_MAX_FILE_BYTES = 8 * 1024 * 1024  # 8 MB

# Total payload cap: raise before writing anything if exceeded.
_MAX_TOTAL_BYTES = 128 * 1024 * 1024  # 128 MB


def extract_doc_examples(source_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Scan ``source_dir`` for document-example families and persist the upload payload.

    Parameters
    ----------
    source_dir:
        Root directory whose immediate subdirectories are family folders.  Each
        family folder MUST contain a ``FAMILY.md`` descriptor and at least one
        additional example file.  Loose files directly under ``source_dir`` are
        ignored with a warning.
    out_dir:
        Directory to write ``families.json`` and ``manifest.json`` into
        (created if missing).

    Returns
    -------
    The manifest dict (also written to ``manifest.json``).

    Raises
    ------
    ValueError
        If no family subdirectories are found, if any family is missing its
        ``FAMILY.md``, if any family has no example files beyond ``FAMILY.md``,
        or if the total payload exceeds 128 MB.
    """
    source = Path(source_dir)
    out = Path(out_dir)

    # Warn about loose files directly under source_dir (not inside a family folder).
    loose = [p for p in source.iterdir() if p.is_file()]
    if loose:
        loose_names = ', '.join(p.name for p in loose)
        logger.warning(
            'Ignoring %d file(s) directly under source_dir (not in a family folder): %s',
            len(loose),
            loose_names,
        )

    family_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if not family_dirs:
        raise ValueError(f'No family subdirectories found under {source}. Each immediate subdirectory should be one document-example family.')

    # --- Validation pass: check FAMILY.md presence + at least one example file ---
    missing_descriptor: list[str] = []
    missing_examples: list[str] = []
    for fdir in family_dirs:
        if not (fdir / FAMILY_DESCRIPTION_FILENAME).is_file():
            missing_descriptor.append(fdir.name)
        else:
            example_files = [p for p in fdir.iterdir() if p.is_file() and p.name != FAMILY_DESCRIPTION_FILENAME]
            if not example_files:
                missing_examples.append(fdir.name)

    if missing_descriptor:
        raise ValueError(f'The following families are missing their required {FAMILY_DESCRIPTION_FILENAME!r} descriptor: ' + ', '.join(missing_descriptor))
    if missing_examples:
        raise ValueError(f'The following families have a {FAMILY_DESCRIPTION_FILENAME!r} but no example files: ' + ', '.join(missing_examples))

    # --- Build upload entries ---
    upload_entries: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    total_bytes = 0

    for fdir in family_dirs:
        family_name = fdir.name
        all_files = sorted(fdir.iterdir())  # deterministic order
        file_entries: list[dict[str, Any]] = []
        skipped = 0

        for fpath in all_files:
            if not fpath.is_file():
                continue
            file_bytes = fpath.read_bytes()
            file_size = len(file_bytes)

            if file_size > _MAX_FILE_BYTES:
                logger.warning(
                    'Skipping %s/%s: file size %d B exceeds the 8 MB per-file cap.',
                    family_name,
                    fpath.name,
                    file_size,
                )
                skipped += 1
                continue

            total_bytes += file_size
            rel_path = fpath.relative_to(fdir).as_posix()  # posix path within family
            file_entries.append(
                {
                    'family': family_name,
                    'path': rel_path,
                    'content_b64': base64.b64encode(file_bytes).decode('ascii'),
                }
            )

        upload_entries.extend(file_entries)

        # example_count = non-FAMILY.md files that were NOT skipped
        example_count = sum(1 for e in file_entries if e['path'] != FAMILY_DESCRIPTION_FILENAME)
        family_summaries.append(
            {
                'name': family_name,
                'example_count': example_count,
                'file_count': len(file_entries),
            }
        )

        logger.debug(
            'Family %r: %d file(s) encoded, %d skipped (>8 MB)',
            family_name,
            len(file_entries),
            skipped,
        )

    # --- Total-size guard (raise BEFORE writing anything) ---
    if total_bytes > _MAX_TOTAL_BYTES:
        raise ValueError(
            f'Total payload is {total_bytes / (1024 * 1024):.1f} MB, which exceeds the 128 MB backend cap. Reduce the number or size of example files.'
        )

    # --- Write output ---
    out.mkdir(parents=True, exist_ok=True)

    (out / FAMILIES_FILENAME).write_text(json.dumps(upload_entries, ensure_ascii=False))

    manifest: dict[str, Any] = {
        'schema_version': SCHEMA_VERSION,
        'family_count': len(family_summaries),
        'families': family_summaries,
        'total_bytes': total_bytes,
        'files': {
            'families': FAMILIES_FILENAME,
        },
    }
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    logger.info(
        'Extracted %d famil%s, %d file(s), %.1f KB total → %s',
        len(family_summaries),
        'y' if len(family_summaries) == 1 else 'ies',
        len(upload_entries),
        total_bytes / 1024,
        out,
    )
    return manifest


__all__ = ['extract_doc_examples']
