"""``plurai stash-doc <file> <label> [--out ID_FILE]``: stash one generation input from disk.

The generation tools take ``*_upload_id`` references so a large document (a PRD, a
``scenario_variables.yaml``) is not re-sent as an inline MCP argument on every call. This command
moves the file from disk to the backend with no context transit and prints its content-addressed
input doc id: identical text gives the identical id, so re-stashing an unchanged file before every
call that needs the id is a server-side no-op, and an edited file mints a new id.

The file goes up through :class:`~plurai.ingest.storage_client.StorageClient` as a storage object,
streamed from disk in parts of at most 512 KiB so every request stays under a 1 MiB request-body
limit, and the object is then named with
``POST /input-docs?label=<label>&storage_object_id=<id>`` and an empty body: the storage upload
API is the backend's one way in for a document's bytes. The backend applies the document checks
(cap, UTF-8, non-blank) and returns the content-addressed id. No lease is minted: a document is not
customer data. The target and bearer resolve as for ``ingest-*``
(:func:`~plurai.ingest.storage_client.resolve_configured_target`); the bearer is never an argument.

stdout carries the id and nothing else, so ``ID=$(plurai stash-doc …)`` is assignable. Everything
else goes to stderr under the ``stash_doc:`` prefix the plugin skills key on: the byte count, each
entry of the backend's ``warnings`` (the skills relay these), the informational ``grounding blocks``
line, every error, and any retry warning the storage client logs. ``--out ID_FILE`` also writes the
id there, atomically and only on success; a failed run removes any file at that path, because a
stale content-addressed id resolves flawlessly to the OLD text. A usage error leaves it alone.

One stash ends within :data:`STASH_DEADLINE_SECONDS`, so a chain of stashes has a known bound. The
by-reference POST is retried once on a transport failure or a 502/503/504, which is safe only
because the id is content-addressed: a duplicate that reaches the backend is a no-op.

EXIT CODES (see :mod:`plurai.exit_codes`)
   0  OK: stashed; the id is on stdout
  66  INPUT_UNREADABLE: file missing, empty or unreadable
  69  TRANSPORT: transport failure, the deadline, a response with no id (including a 5xx), or an upload
      refused after its object was created: state unknown, the file is fine, and a fresh stash
      can succeed; the caller may fall back to the inline text parameter
  12  ROUTE_MISSING: the backend predates this stash (no ``POST /input-docs``, no ``/storage``, no part uploads
      on ``/storage``, or an ``/input-docs`` that ignores ``storage_object_id``): use the MCP
      ``stash_input_doc`` fallback
  13  JOB_FAILED: the backend refused for credit (a 402 naming a credit kind, on the upload or the stash); the kind's
      sentence is on stderr, not the gateway's text, and nothing was stashed
  77  CREDENTIALS: no credential configured, or the bearer was rejected (401/403)
  17  BACKEND_REFUSED: the document was refused (over the backend's cap, not UTF-8, or blank, all checked
      before any request; or a 413 from the backend or its proxy): fix the file, and do not fall
      back to inline, which meets the same refusal through the model's context
  64  USAGE: including a label outside ``[A-Za-z0-9._-]`` (it rides the query string
      unencoded and is a server-side discovery key: a mangled label stashes but is invisible to
      the backend's label lookups)
"""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, TextIO

import httpx
from pydantic import ValidationError

from plurai import client_handshake, exit_codes
from plurai._http_retry import RATE_LIMITED_STATUS
from plurai.customer_safe import describe_failure, response_json, safe_text
from plurai.ingest._command import client as authed_client
from plurai.ingest._command import TRANSIENT_STATUS_CODES, CommandError, credit_stop, prefixed_warnings, send_with_retry, write_atomic
from plurai.ingest.storage_client import (
    MAX_OBJECT_NAME_LENGTH,
    PartUploadUnsupportedError,
    StorageClient,
    StorageConfigError,
    StorageCreditStopError,
    StorageUploadError,
    resolve_configured_target,
)


LABEL_PATTERN = re.compile(r'[A-Za-z0-9._-]+')

#: The largest document the backend stashes. The backend enforces it only after the upload has
#: completed, so the client refuses an over-cap file first. The route names the cap only in its
#: contract prose, so the client holds this literal.
MAX_INPUT_DOC_BYTES = 8 * 1024 * 1024
#: The whole stash, upload and naming included.
STASH_DEADLINE_SECONDS = 60.0
#: Each by-reference POST attempt: the backend reads, decodes and hashes the stored document inside it.
STASH_ATTEMPT_SECONDS = 14.0
#: One retry, one second later: two attempts fit the deadline after a 30 s upload.
_STASH_RETRY_DELAY_SECONDS = 1.0

_READ_CHUNK_BYTES = 1024 * 1024
_PREFIX = 'stash_doc'
_FALLBACK_HINT = 'fall back to the MCP stash_input_doc tool'


class StashError(CommandError):
    """A classified failure of the stash, coded by the module's exit table."""


def _say(message: str, err: TextIO) -> None:
    print(f'{_PREFIX}: {message}', file=err)


def scan_document(path: Path) -> tuple[int, str]:
    """Size and SHA-256 of ``path`` from one bounded-memory pass that also applies the backend's
    own refusals (over :data:`MAX_INPUT_DOC_BYTES`, UTF-8, non-blank) before any request, so a
    bad file costs no upload. A document is at least one byte; the storage contract creates
    objects of ``contentLength > 0``, so a zero-byte file never reaches it.

    Raises:
        StashError: 1 when the file is missing, empty or unreadable; 5 when it is over the cap,
            not UTF-8 or blank.
    """
    if not path.is_file() or not os.access(path, os.R_OK):
        raise StashError(exit_codes.INPUT_UNREADABLE, f'no readable non-empty file at {path}')
    decoder = codecs.getincrementaldecoder('utf-8')()
    hasher = hashlib.sha256()
    size = 0
    has_text = False
    try:
        # Before the read pass, so an oversized file is not hashed.
        if (declared := path.stat().st_size) > MAX_INPUT_DOC_BYTES:
            raise _over_cap(declared)
        with path.open('rb') as handle:
            while chunk := handle.read(_READ_CHUNK_BYTES):
                # The decoder may hold the head of a split multibyte character from the last read.
                pending = len(decoder.getstate()[0])
                try:
                    text = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as error:
                    raise _not_utf8(path, f'{error.reason} at byte {size - pending + error.start}') from error
                has_text = has_text or bool(text.strip())
                hasher.update(chunk)
                size += len(chunk)
        if decoder.getstate()[0]:
            raise _not_utf8(path, f'truncated character at byte {size - len(decoder.getstate()[0])}')
    except OSError as error:
        raise StashError(exit_codes.INPUT_UNREADABLE, f'no readable non-empty file at {path} ({error.strerror or error})') from error
    if size == 0:
        raise StashError(exit_codes.INPUT_UNREADABLE, f'no readable non-empty file at {path}')
    if size > MAX_INPUT_DOC_BYTES:  # the file grew during the pass
        raise _over_cap(size)
    if not has_text:
        raise StashError(exit_codes.BACKEND_REFUSED, f'backend would refuse the document (HTTP 400): {path} is empty or whitespace-only. Nothing was sent.')
    return size, hasher.hexdigest()


def _over_cap(size: int) -> StashError:
    return StashError(
        exit_codes.BACKEND_REFUSED,
        f'backend would refuse the document (HTTP 413): input doc too large ({size} bytes); the cap is {MAX_INPUT_DOC_BYTES}. '
        'Stash the document that actually steers generation; do not send it inline, which meets the same limit. Nothing was sent.',
    )


def _not_utf8(path: Path, reason: str) -> StashError:
    return StashError(
        exit_codes.BACKEND_REFUSED,
        f'backend would refuse the document (HTTP 400): {path} is not UTF-8 text (a PRD / scenario_variables.yaml, '
        f'not a binary): {reason}. Fix the file; nothing was sent.',
    )


def _storage_exit(error: StorageUploadError) -> int:
    """The exit code for a storage refusal: on the create it is about the document (or a backend
    with no object store); after it, a 413 is the transport refusing the size and anything else
    is the upload's fault, which a fresh stash can fix."""
    status, on_create = error.status_code, error.endpoint == '/storage/objects'
    if status in (401, 403):
        return exit_codes.CREDENTIALS
    if status == client_handshake.UPGRADE_STATUS:
        return exit_codes.UPGRADE_REQUIRED
    if on_create and status in (404, 405):
        return exit_codes.ROUTE_MISSING
    if status == 413 or (on_create and status in (400, 409, 422)):
        return exit_codes.BACKEND_REFUSED
    return exit_codes.TRANSPORT


_THROTTLE_HINT = '; the Plurai backend is rate limiting this client — wait a minute and stash again'


def _throttle_hint(error: Exception) -> str:
    """What to do about a 429 that outlasted the retries, else nothing."""
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == RATE_LIMITED_STATUS:
        return _THROTTLE_HINT
    return ''


async def _upload(client: StorageClient, path: Path, size: int, checksum: str) -> str:
    """Create the storage object, send its content, return the object id."""
    try:
        object_id, limits = await client.create_storage_object(
            name=path.name[:MAX_OBJECT_NAME_LENGTH], content_type='text/plain', content_length=size, checksum=checksum
        )
        await client.upload_content(object_id, path, size, limits)
    except PartUploadUnsupportedError as error:
        raise StashError(exit_codes.ROUTE_MISSING, f'{error}; {_FALLBACK_HINT}') from error
    except StorageCreditStopError as error:
        raise StashError(exit_codes.JOB_FAILED, str(error)) from error
    except StorageUploadError as error:
        code = _storage_exit(error)
        hint = {
            exit_codes.ROUTE_MISSING: f'; this backend has no /storage object store and predates this stash — {_FALLBACK_HINT}',
            exit_codes.BACKEND_REFUSED: '; do not send the document inline, which meets the same limit',
            exit_codes.TRANSPORT: '; the file is fine and a fresh stash can succeed',
        }.get(code, '')
        raise StashError(code, f'{error}{hint}') from error
    except (httpx.HTTPError, ValidationError, ValueError) as error:
        # A 5xx or a throttle that outlasted the retries, a transport failure, or a create answer this client cannot read.
        raise StashError(exit_codes.TRANSPORT, f'storage upload to {client.base_url} failed: {error}{_throttle_hint(error)}') from error
    return object_id


async def _post_by_reference(http: httpx.AsyncClient, label: str, object_id: str) -> httpx.Response:
    """``POST /input-docs?label=…&storage_object_id=…`` with an empty body; the last response,
    whatever its status."""
    params = {'label': label, 'storage_object_id': object_id}
    try:
        return await send_with_retry(
            lambda: http.post('/input-docs', params=params, content=b''),
            attempts=2,
            delay=_STASH_RETRY_DELAY_SECONDS,
            timeout=STASH_ATTEMPT_SECONDS,
            retry_statuses=TRANSIENT_STATUS_CODES,
        )
    except (httpx.HTTPError, TimeoutError) as error:
        raise StashError(exit_codes.TRANSPORT, f'transport failure reaching {http.base_url}input-docs: {error!r}') from error


def classify_stash(response: httpx.Response, object_id: str) -> dict[str, Any]:
    """The parsed success body of a by-reference stash, or the classified failure.

    The backend answers an unknown or incomplete object with a 404/409 naming the storage object: the
    route answered, so that is a transport-class failure, not a missing route. A 400 about an
    empty body means the route read the empty body instead of the object, so it does not
    support ``storage_object_id``; the file itself holds text (checked locally).

    Raises:
        StashError: classified per the module's exit-code table.
    """
    status = response.status_code
    if (stop := credit_stop(response)) is not None:
        raise StashError(stop.code, str(stop))
    detail = describe_failure(response)
    if status in (401, 403):
        raise StashError(exit_codes.CREDENTIALS, f'bearer rejected (HTTP {status})')
    if status == client_handshake.UPGRADE_STATUS:
        raise StashError(exit_codes.UPGRADE_REQUIRED, client_handshake.upgrade_message_for(response))
    if status == RATE_LIMITED_STATUS:
        raise StashError(exit_codes.TRANSPORT, f'backend throttled the stash (HTTP {status}){_THROTTLE_HINT}')
    if status in (404, 409) and 'storage object' in detail:
        raise StashError(exit_codes.TRANSPORT, f'backend could not stash storage object {object_id} (HTTP {status}): {detail}')
    if status in (404, 405) or (status == 400 and 'empty' in detail):
        raise StashError(
            exit_codes.ROUTE_MISSING,
            f'POST /input-docs by storage_object_id not on this backend (HTTP {status}: {detail}) — {_FALLBACK_HINT}',
        )
    if status in (400, 413):
        raise StashError(exit_codes.BACKEND_REFUSED, f'backend refused the document (HTTP {status}): {detail}')
    body = response_json(response)
    if not isinstance(body, dict) or not isinstance(body.get('input_doc_id'), str) or not body['input_doc_id']:
        raise StashError(exit_codes.TRANSPORT, f'HTTP {status} with no input_doc_id in the body: {detail}')
    return body


def report_stash(body: dict[str, Any], err: TextIO) -> None:
    """The stderr half of a successful stash: bytes, every warning, then the block sizes.

    The size line is decoration on a stash that already succeeded, so a payload this cannot
    render is skipped rather than allowed to turn the stash into a failure.
    """
    _say(f'stashed {body.get("bytes", 0)} bytes', err)
    warnings = body.get('warnings')
    for warning in warnings if isinstance(warnings, list) else []:
        _say(f'WARNING: {safe_text(str(warning))}', err)
    try:
        sizes = ' · '.join(f'{block["name"]} {block["chars"]:,}/{block["budget"]:,}' for block in body.get('excerpt_sizes') or [])
    except Exception:  # noqa: BLE001 - see the docstring
        sizes = ''
    if sizes:
        _say(f'grounding blocks — {sizes} chars', err)


def write_id_file(out: Path, doc_id: str) -> None:
    """Write ``doc_id`` to ``out`` atomically: a temp file beside it, then a rename.

    Raises:
        StashError: 2, naming the id, since the document was stashed.
    """
    try:
        write_atomic(out, f'{doc_id}\n'.encode())
    except OSError as error:
        raise StashError(exit_codes.TRANSPORT, f'could not write the id to --out {out} ({error.strerror}); the document WAS stashed as {doc_id}') from error


async def stash(path: Path, label: str, err: TextIO, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Scan, resolve, upload and name the document; return the input doc id.

    ``transport`` is the test seam for both the storage client and the by-reference POST.

    Raises:
        StashError: classified per the module's exit-code table.
    """
    size, checksum = scan_document(path)
    try:
        base, token = resolve_configured_target()
    except StorageConfigError as error:
        raise StashError(exit_codes.CREDENTIALS, f'ERROR: {error}') from error
    async with StorageClient(base, token, transport=transport) as client:
        object_id = await _upload(client, path, size, checksum)
    async with authed_client(base, token, STASH_ATTEMPT_SECONDS, transport) as http:
        response = await _post_by_reference(http, label, object_id)
    body = classify_stash(response, object_id)
    report_stash(body, err)
    return body['input_doc_id']


def run_stash_doc(
    file: str,
    label: str,
    out: str | None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai stash-doc`` command: returns its exit code.

    The id reaches stdout only after ``--out`` is written, so a run that exits non-zero has printed
    no id anywhere a caller could capture it.
    """
    err = stderr if stderr is not None else sys.stderr
    if out == '':
        _say('usage: plurai stash-doc <abs-file-path> <label> [--out <id-file>]  (--out needs a path)', err)
        return exit_codes.USAGE
    if not LABEL_PATTERN.fullmatch(label):
        _say(f'label must match [A-Za-z0-9._-]+ (it is a server-side discovery key and rides the URL unencoded), got: {label}', err)
        return exit_codes.USAGE
    try:
        with prefixed_warnings(_PREFIX, err):
            doc_id = asyncio.run(asyncio.wait_for(stash(Path(file), label, err, transport), STASH_DEADLINE_SECONDS))
        if out is not None:
            write_id_file(Path(out), doc_id)
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        if isinstance(error, StashError):
            code, message = error.code, str(error)
        else:
            code = exit_codes.TRANSPORT
            message = f'the stash did not end within {STASH_DEADLINE_SECONDS:.0f}s' if isinstance(error, TimeoutError) else f'unexpected error: {error!r}'
        _say(message, err)
        if out:
            try:
                Path(out).unlink(missing_ok=True)
            except OSError as removal:
                _say(f'could not remove the stale id file {out} ({removal.strerror}); do NOT use the id in it', err)
        return code
    print(doc_id, file=stdout if stdout is not None else sys.stdout)
    return exit_codes.OK


__all__ = ['StashError', 'run_stash_doc', 'stash']
