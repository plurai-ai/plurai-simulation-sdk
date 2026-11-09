"""``plurai fetch policies`` and ``plurai fetch artifact``: backend-owned files to disk, never through a
model's context.

``fetch policies <eval_set_id> --out F`` syncs a local policy mirror from
``GET /eval-sets/{id}/policies.yaml``. That route serves whatever is on disk, and during an open
edit session that is the session's working DRAFT, byte for byte indistinguishable from the promoted
set. The backend says which in the policies-state response header (``STATE_HEADER``; ``promoted``, ``draft``, or ``unknown``
when it has no session store) and names the session in the edit-session header (``SESSION_HEADER``); this command is
the one reader of that pair. A draft is refused and ``F`` left untouched, because a draft is the
content that must not become the file an operator reviews; ``--allow-draft`` writes it for a caller
that wants the draft on purpose (a scratch copy). ``unknown`` and a missing header (a backend that
predates it, reported as ``absent``) both write, with a WARNING. stdout is exactly
``state=<promoted|draft|unknown|absent>``, on exit 0 and on the draft refusal; an unrecognised
header value reads as ``unknown``. An empty 2xx body is refused: no promoted eval set is empty,
and writing one would leave a zero-byte mirror through the success path.

``fetch artifact <id> <relpath>... (--dir D | --out F)`` downloads files through
``GET /artifacts/{id}/files/{relpath}``: each to ``D/<relpath>``, or the single one to ``F``.
stdout is one line per file, ``wrote <path>`` or ``missing <relpath>``, so a caller can fall back
per file; ``<path>`` is ``--out`` verbatim or ``<dir>/<relpath>``, never normalised. The route
answers a file it does not have with a 404 carrying its own ``{"error": ...}`` body; a 404 or 405
without that body is a backend with no such route.

A binary document (a PDF a scenario hands the agent, say) is not text, so the artifact route answers
it with a one-line placeholder naming its size. ``fetch artifact`` recognises that line and fetches
the real bytes from ``GET /documents/{artifact_id}/{scenario}/{path}``, the route that serves them,
for a relpath of the form ``<scenario>/documents/<path>``. The bytes are written only when their
length equals the size the placeholder named; a placeholder that cannot be resolved is an error
(exit TRANSPORT, ROUTE_MISSING or CREDENTIALS), never a file.

Every file is written through a temp file beside its destination and a rename, so a failed fetch
never truncates a good file. Each GET is retried twice, one second apart, on a transport failure
or a gateway status (502/503/504); each attempt ends within :data:`ATTEMPT_SECONDS`. Redirects are
not followed: a login page at the end of a chain would be written as the file.

Target and bearer resolve from the configuration (:func:`~plurai.ingest._command.resolve_target`),
never from an argument. Nothing here checks WHICH backend answers: a caller that also holds an
MCP session runs its split-brain check first and exports the URL it checked as
``PLURAI_URL``, or a sibling deployment's file lands on the local one. Diagnostics go
to stderr under ``fetch_policies:`` or ``fetch_artifact:``.

EXIT CODES — fetch policies (see :mod:`plurai.exit_codes`)
   0  OK: written to ``--out`` (with a WARNING for ``unknown`` and ``absent``)
  77  CREDENTIALS: nothing is configured, or the bearer was rejected (401/403); ``--out`` untouched
  65  REFUSED: the configured target is not a usable URL; ``--out`` untouched
  69  TRANSPORT: transport failure, a non-2xx, an empty body, or ``--out`` not writable; ``--out`` untouched
  10  UPGRADE_REQUIRED: the backend refuses this client version (426); ``--out`` untouched
  13  JOB_FAILED: the backend refused for credit (a 402 naming a credit kind); ``--out`` untouched
  16  NOT_READY: the backend says DRAFT and ``--allow-draft`` was not given; ``--out`` untouched: end the
      edit session, then re-run
  64  USAGE

EXIT CODES — fetch artifact
   0  OK: every file written
  15  NOT_FOUND: some file is missing (the route's own 404), the others were written
  77  CREDENTIALS and 65 REFUSED, 10 UPGRADE_REQUIRED: as for ``fetch policies``; nothing was fetched
  69  TRANSPORT: transport failure, a 5xx or other status, a destination not writable, or a binary document
      whose bytes cannot be fetched or do not match the size the backend named: stop, nothing
      after it was fetched, and the destination is untouched
  13  JOB_FAILED: the backend refused for credit (a 402 naming a credit kind); the kind's sentence is on stderr
  12  ROUTE_MISSING: a 404/405 without the route's body: use the MCP ``fetch_artifact`` tool
  64  USAGE: including a relpath that is absolute or has an empty, ``.`` or ``..``
      segment
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import TextIO
from urllib.parse import quote

import httpx

from plurai import client_handshake, exit_codes
from plurai.customer_safe import response_json
from plurai.ingest._command import (
    TRANSIENT_STATUS_CODES,
    CommandError,
    client,
    credit_stop,
    describe_failure,
    get_with_retry,
    resolve_target,
    say,
    upgrade_required,
    write_atomic,
)


STATE_HEADER = 'X-Plurai-Policies-State'
SESSION_HEADER = 'X-Plurai-Edit-Session'
STATES = ('promoted', 'draft', 'unknown')
ATTEMPT_SECONDS = 60.0
RETRY_DELAY_SECONDS = 1.0
ATTEMPTS = 3

#: The body the artifact route serves in place of a binary document (the dash may be an em dash,
#: an en dash or a hyphen). Group 1 is the document's size in bytes.
BINARY_PLACEHOLDER = re.compile(r'<binary document\s*[\u2014\u2013-]\s*(\d+) bytes; fetch via the /documents URL channel>')
#: A placeholder is one short line; a larger body is a text file, never inspected.
PLACEHOLDER_MAX_BYTES = 512


async def _get(http: httpx.AsyncClient, path: str) -> httpx.Response:
    """The last response for ``GET path``, whatever its status.

    Raises:
        CommandError: 2 on a transport failure that outlasts the retries.
    """
    try:
        return await get_with_retry(http, path, attempts=ATTEMPTS, delay=RETRY_DELAY_SECONDS, timeout=ATTEMPT_SECONDS, retry_statuses=TRANSIENT_STATUS_CODES)
    except (httpx.HTTPError, TimeoutError) as error:
        raise CommandError(exit_codes.TRANSPORT, f'transport failure reaching {http.base_url.join(path)} ({ATTEMPTS} attempts): {error!r}') from error


def _write(dest: Path, content: bytes) -> None:
    """Raises:
    CommandError: 2, with ``dest`` untouched.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(dest, content)
    except OSError as error:
        raise CommandError(exit_codes.TRANSPORT, f'could not write {dest} ({error.strerror}); it was left untouched') from error


def _header(response: httpx.Response, name: str) -> str:
    """The last value of a header, trimmed; names match case-insensitively, as proxies re-case them."""
    values = response.headers.get_list(name)
    return values[-1].strip() if values else ''


async def fetch_policies(eval_set_id: str, out: Path, allow_draft: bool, err: TextIO, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """Fetch, classify and write; return the state word for stdout.

    Raises:
        CommandError: per the ``fetch policies`` exit table.
    """
    base, token = resolve_target()
    async with client(base, token, ATTEMPT_SECONDS, transport) as http:
        response = await _get(http, f'/eval-sets/{quote(eval_set_id, safe="")}/policies.yaml')
    url = response.url
    if response.status_code == client_handshake.UPGRADE_STATUS:
        raise upgrade_required(response)
    if response.status_code in (401, 403):
        raise CommandError(exit_codes.CREDENTIALS, f'bearer rejected (HTTP {response.status_code}) at {url}')
    if (stop := credit_stop(response)) is not None:
        raise stop
    if not 200 <= response.status_code < 300:
        raise CommandError(exit_codes.TRANSPORT, f'HTTP {response.status_code} from {url}: {describe_failure(response)}')
    if not response.content:
        raise CommandError(
            exit_codes.TRANSPORT,
            f'HTTP {response.status_code} but the body is EMPTY: refusing to write a zero-byte policy set over {out}. Nothing was changed.',
        )
    raw = _header(response, STATE_HEADER)
    state = raw if raw in STATES else ('absent' if not raw else 'unknown')
    session = _header(response, SESSION_HEADER) or '<unnamed>'
    if state == 'draft' and not allow_draft:
        raise CommandError(
            exit_codes.NOT_READY,
            f'the backend says DRAFT: edit session {session} is still OPEN on eval set {eval_set_id} and nothing has been promoted. '
            f'{out} was left untouched; end the session, then re-run.',
        )
    if state == 'draft':
        say('fetch_policies', f'NOTE: these are the DRAFT bytes of open edit session {session}, not the promoted set.', err)
    elif state == 'unknown':
        say(
            'fetch_policies',
            f'WARNING: {url} reports {STATE_HEADER}={raw}: this server cannot tell a draft from the promoted set. '
            'Writing the bytes anyway; confirm no edit session is open before trusting this file.',
            err,
        )
    elif state == 'absent':
        say(
            'fetch_policies',
            f'WARNING: {url} sent no {STATE_HEADER} header (an older backend), so whether these bytes are promoted is unverified. '
            'Writing them anyway; make sure end_edit_session has COMMITTED first.',
            err,
        )
    _write(out, response.content)
    return state


def run_fetch_policies(
    eval_set_id: str,
    out: str,
    allow_draft: bool,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai fetch policies`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    stdout = stdout if stdout is not None else sys.stdout
    if not eval_set_id.strip() or not out:
        say('fetch_policies', 'usage: plurai fetch policies <eval_set_id> --out <dest> [--allow-draft]', err)
        return exit_codes.USAGE
    try:
        state = asyncio.run(fetch_policies(eval_set_id, Path(out), allow_draft, err, transport))
    except CommandError as error:
        say('fetch_policies', str(error), err)
        if error.code == exit_codes.NOT_READY:
            print('state=draft', file=stdout)
        return error.code
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        say('fetch_policies', f'unexpected error: {error!r}; {out} was left untouched', err)
        return exit_codes.TRANSPORT
    print(f'state={state}', file=stdout)
    return exit_codes.OK


def artifact_destinations(relpaths: list[str], directory: str | None, out: str | None) -> list[str]:
    """Where each relpath lands, spelled as the caller spelled it: ``--out`` verbatim, else
    ``<dir>/<relpath>`` joined with one ``/`` and no normalising. The ``wrote`` line prints this
    string, and a caller matches it against the path it built the same way (a trailing slash on
    the directory, a ``./`` prefix), so a normalised spelling would read as a missing file.

    Raises:
        CommandError: 64 for ``--out`` with other than one relpath, or a relpath that is absolute,
        has an empty, ``.`` or ``..`` segment, or a backslash.
    """
    for rel in relpaths:
        # Every segment must name something, so the file the backend serves is the file written.
        if '\\' in rel or any(segment in ('', '.', '..') for segment in rel.split('/')):
            raise CommandError(exit_codes.USAGE, f'relpath must be a relative path inside the artifact, got {rel!r}')
    if out is not None:
        if len(relpaths) != 1:
            raise CommandError(exit_codes.USAGE, f'--out takes exactly one relpath, got {len(relpaths)}; use --dir for several')
        return [out]
    return [f'{directory}/{rel}' if directory else rel for rel in relpaths]


def _missing_error(response: httpx.Response) -> str | None:
    """The route's own ``error`` for a file it does not have, else ``None``."""
    body = response_json(response)
    return body['error'] if response.status_code == 404 and isinstance(body, dict) and isinstance(body.get('error'), str) else None


def _classify_artifact_file(response: httpx.Response, rel: str) -> None:
    """Raises:
    CommandError: per the ``fetch artifact`` exit table, for a response that is neither a
    file nor the route's own 404.
    """
    status = response.status_code
    if 200 <= status < 300:
        return
    if status in (401, 403):
        raise CommandError(exit_codes.CREDENTIALS, f'bearer rejected (HTTP {status}) at {response.url}')
    if status == client_handshake.UPGRADE_STATUS:
        raise upgrade_required(response)
    if (stop := credit_stop(response)) is not None:
        raise stop
    if status in (404, 405):
        raise CommandError(exit_codes.ROUTE_MISSING, f'{response.url} answered {status} without the route body: this backend has no artifact file route')
    raise CommandError(exit_codes.TRANSPORT, f'HTTP {status} from {response.url} for {rel}: {describe_failure(response)}')


def _placeholder_size(content: bytes) -> int | None:
    """The size a binary-document placeholder body names, or ``None`` for any other body."""
    if len(content) > PLACEHOLDER_MAX_BYTES:
        return None
    try:
        match = BINARY_PLACEHOLDER.fullmatch(content.decode('utf-8').strip())
    except UnicodeDecodeError:
        return None
    return int(match.group(1)) if match else None


def _document_path(artifact_id: str, rel: str) -> str | None:
    """The ``/documents`` route path serving ``<scenario>/documents/<path>`` of the artifact, else ``None``."""
    scenario, marker, document = rel.partition('/documents/')
    if not marker or not scenario or not document:
        return None
    return f'/documents/{quote(artifact_id, safe="")}/{quote(scenario, safe="")}/{quote(document, safe="/")}'


async def _fetch_binary_document(http: httpx.AsyncClient, artifact_id: str, rel: str, size: int) -> bytes:
    """The real bytes of the binary document ``rel``, whose artifact body was a placeholder naming ``size``.

    Raises:
        CommandError: 2 when the document cannot be located, fetched or does not match ``size``;
        4 when the bearer is rejected; 3 when the backend has no ``/documents`` route.
    """
    path = _document_path(artifact_id, rel)
    if path is None:
        raise CommandError(
            exit_codes.TRANSPORT,
            f'{rel} is a binary document ({size} bytes) outside <scenario>/documents/<path>, so its bytes cannot be located; nothing was written',
        )
    response = await _get(http, path)
    status = response.status_code
    if status in (401, 403):
        raise CommandError(exit_codes.CREDENTIALS, f'bearer rejected (HTTP {status}) at {response.url}')
    if status == client_handshake.UPGRADE_STATUS:
        raise upgrade_required(response)
    if (stop := credit_stop(response)) is not None:
        raise stop
    if status in (404, 405) and _missing_error(response) is None:
        raise CommandError(
            exit_codes.ROUTE_MISSING, f'{response.url} answered {status} without the route body: no /documents route here; {rel} was not written'
        )
    if not 200 <= status < 300:
        raise CommandError(
            exit_codes.TRANSPORT, f'HTTP {status} from {response.url} for the binary document {rel}: {describe_failure(response)}; it was not written'
        )
    if len(response.content) != size:
        raise CommandError(
            exit_codes.TRANSPORT,
            f'binary document {rel}: the backend named {size} bytes but {response.url} served {len(response.content)}; it was not written',
        )
    return response.content


async def fetch_artifact(
    artifact_id: str,
    relpaths: list[str],
    dests: list[str],
    out: TextIO,
    err: TextIO,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """Fetch and write each file in order, printing its line; return 0 or 1.

    Raises:
        CommandError: 1 for a refused target; 2, 3 or 4 at the first file that cannot be
        classified as written or missing.
    """
    code = exit_codes.OK
    base, token = resolve_target()
    async with client(base, token, ATTEMPT_SECONDS, transport) as http:
        for rel, dest in zip(relpaths, dests, strict=True):
            response = await _get(http, f'/artifacts/{quote(artifact_id, safe="")}/files/{quote(rel, safe="/")}')
            if (missing := _missing_error(response)) is not None:
                say('fetch_artifact', f'{rel}: {missing}', err)
                print(f'missing {rel}', file=out, flush=True)
                code = exit_codes.NOT_FOUND
                continue
            _classify_artifact_file(response, rel)
            content = response.content
            if (size := _placeholder_size(content)) is not None:
                content = await _fetch_binary_document(http, artifact_id, rel, size)
            _write(Path(dest), content)
            print(f'wrote {dest}', file=out, flush=True)
    return code


def run_fetch_artifact(
    artifact_id: str,
    relpaths: list[str],
    directory: str | None,
    out: str | None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai fetch artifact`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    try:
        if not artifact_id.strip() or not relpaths or out == '' or directory == '':
            raise CommandError(exit_codes.USAGE, 'usage: plurai fetch artifact <artifact_id> <relpath>... (--dir DIR | --out FILE)')
        dests = artifact_destinations(relpaths, directory, out)
        return asyncio.run(fetch_artifact(artifact_id, relpaths, dests, stdout if stdout is not None else sys.stdout, err, transport=transport))
    except CommandError as error:
        say('fetch_artifact', str(error), err)
        return error.code
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        say('fetch_artifact', f'unexpected error: {error!r}', err)
        return exit_codes.TRANSPORT


__all__ = ['artifact_destinations', 'fetch_artifact', 'fetch_policies', 'run_fetch_artifact', 'run_fetch_policies']
