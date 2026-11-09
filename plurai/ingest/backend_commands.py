"""``plurai backend identity`` and ``plurai backend probe``: two read-only questions about the backend.

``identity`` asks which deployment answers: ``GET /backend-identity`` returns the same flat object
as the MCP ``backend_identity()`` tool (``instance_id``, ``started_at``, ``hostname``, ``build_id``,
``source_id``, ``organization_id``, ``workspace_id``, ``storage_root``), and stdout carries it as one
line of JSON for the caller's split-brain comparison. The comparison stays with the caller: only it
holds the MCP side. A transport failure is retried once, one second later, because a single blip
would otherwise disarm the one check that tells two backends apart; an answer of any status is
final.

``probe`` asks whether the backend is there: it prints exactly the HTTP code of ``GET <path>``
(``/healthz`` by default), ``000`` when nothing answered, so ``code=$(plurai backend probe)`` is
assignable. It follows up to :data:`MAX_REDIRECTS` redirects, because an edge that upgrades http
to https answers 301/308 for a healthy backend. A 2xx counts only when the final URL still ends in
the requested path; an SSO proxy redirects to a login page that answers 200 with no backend behind
it, and that prints as ``3xx``. Every other answer, a 401 from an auth layer, a 404 from another
service on the port, a 502 from a proxy with no upstream, proves only that something answered.

Target and bearer resolve from the configuration (:func:`~plurai.ingest._command.resolve_target`),
never from an argument. Diagnostics go to stderr under ``backend_identity:`` or ``backend_probe:``.

EXIT CODES — backend identity (see :mod:`plurai.exit_codes`)
   0  OK: the identity is on stdout
  77  CREDENTIALS: nothing is configured, or the bearer was rejected (401/403)
  65  REFUSED: the configured target is not a usable URL
  13  JOB_FAILED: the backend refused for credit (a 402 naming a credit kind); the kind's sentence is on stderr
  69  TRANSPORT: no response, after one retry; an unexpected client error; or any other status
  12  ROUTE_MISSING: no identity route (404/405), or a 2xx that is not a JSON object carrying ``instance_id``:
      this backend predates the route
  10  UPGRADE_REQUIRED: the backend refuses this client version (426)
  64  USAGE

EXIT CODES — backend probe
   0  OK: 2xx on the requested path
  77  CREDENTIALS: nothing is configured; nothing is printed
  65  REFUSED: the configured target is not a usable URL; nothing is printed
  69  TRANSPORT: anything else: ``000``, a non-2xx, or ``3xx`` for a 2xx that landed off the path
  64  USAGE: including a path without a leading ``/``

An unexpected client error in ``probe`` prints ``000``, like no answer.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import TextIO

import httpx

from plurai import client_handshake, exit_codes
from plurai._endpoints import CREDENTIAL_FIX
from plurai.customer_safe import project_value, response_json
from plurai.ingest._command import (
    CommandError,
    client,
    credit_stop,
    describe_failure,
    get_with_retry,
    resolve_target,
    say,
    send_with_retry,
    upgrade_required,
)


IDENTITY_PATH = '/backend-identity'
#: The field that makes a body an identity: the other fields are optional across versions.
IDENTITY_KEY = 'instance_id'
HEALTHZ_PATH = '/healthz'
#: The split-brain check is a preflight on every flow that uses it, so its budget is the tightest:
#: each attempt ends within this, connect included.
IDENTITY_ATTEMPT_SECONDS = 5.0
#: One retry, this long after the first attempt: back-to-back attempts land in the same sub-second blip.
IDENTITY_RETRY_DELAY_SECONDS = 1.0
PROBE_TIMEOUT_SECONDS = 5.0


async def probe(http: httpx.AsyncClient, path: str, timeout: float) -> str:
    """The HTTP code of ``GET path`` by the rule in the module docstring, as the caller prints it.
    ``timeout`` bounds the whole exchange, redirects included; a chain past
    :data:`~plurai.ingest._command.MAX_REDIRECTS` is no answer."""
    try:
        response = await send_with_retry(lambda: http.get(path, follow_redirects=True), attempts=1, delay=0.0, timeout=timeout)
    except (httpx.HTTPError, TimeoutError):
        return '000'
    code = response.status_code
    if 200 <= code < 300 and not response.url.path.rstrip('/').endswith(path.split('?', 1)[0].rstrip('/')):
        return '3xx'
    return str(code)


def is_alive(code: str) -> bool:
    """Up means a 2xx on the path; see the module docstring for why nothing else is."""
    return len(code) == 3 and code.startswith('2')


def classify_identity(response: httpx.Response) -> dict:
    """The identity object in a ``GET /backend-identity`` answer, or the classified failure.

    Raises:
        CommandError: per the ``backend identity`` exit table.
    """
    status = response.status_code
    if status in (401, 403):
        raise CommandError(exit_codes.CREDENTIALS, f'bearer rejected (HTTP {status}): the token does not match the backend at {response.url}; {CREDENTIAL_FIX}')
    if status == client_handshake.UPGRADE_STATUS:
        raise upgrade_required(response)
    if (stop := credit_stop(response)) is not None:
        raise stop
    if status in (404, 405):
        raise CommandError(exit_codes.ROUTE_MISSING, f'no identity route at {response.url} (HTTP {status}): this backend predates it')
    if not 200 <= status < 300:
        raise CommandError(exit_codes.TRANSPORT, f'HTTP {status} from {response.url}: {describe_failure(response)}')
    body = response_json(response)
    if not isinstance(body, dict) or not body.get(IDENTITY_KEY):
        raise CommandError(
            exit_codes.ROUTE_MISSING, f'HTTP {status} from {response.url} is not an identity object with {IDENTITY_KEY}: {describe_failure(response)}'
        )
    return body


async def fetch_identity(base: str, token: str | None, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """``GET /backend-identity``, retried once on a transport failure.

    Raises:
        CommandError: per the ``backend identity`` exit table.
    """
    async with client(base, token, IDENTITY_ATTEMPT_SECONDS, transport) as http:
        try:
            response = await get_with_retry(http, IDENTITY_PATH, attempts=2, delay=IDENTITY_RETRY_DELAY_SECONDS, timeout=IDENTITY_ATTEMPT_SECONDS)
        except (httpx.HTTPError, TimeoutError) as error:
            raise CommandError(
                exit_codes.TRANSPORT, f'{base}{IDENTITY_PATH} did not respond (two attempts, {IDENTITY_RETRY_DELAY_SECONDS:.0f}s apart): {error!r}'
            ) from error
    return classify_identity(response)


def run_backend_identity(
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai backend identity`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    try:
        base, token = resolve_target()
        identity = asyncio.run(fetch_identity(base, token, transport))
    except CommandError as error:
        say('backend_identity', str(error), err)
        return error.code
    except Exception as error:  # noqa: BLE001 - an uncaught exception would exit 1, the policies-failed row
        say('backend_identity', f'unexpected error: {error!r}', err)
        return exit_codes.TRANSPORT
    print(json.dumps(project_value(identity), ensure_ascii=False), file=stdout if stdout is not None else sys.stdout)
    return exit_codes.OK


async def _probe_once(base: str, token: str | None, path: str, timeout: float, transport: httpx.AsyncBaseTransport | None) -> str:
    async with client(base, token, timeout, transport) as http:
        return await probe(http, path, timeout)


def run_backend_probe(
    path: str,
    timeout: float,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    """The ``plurai backend probe`` command: returns its exit code."""
    err = stderr if stderr is not None else sys.stderr
    if not path.startswith('/') or timeout <= 0:
        say('backend_probe', f'usage: plurai backend probe [--path /P] [--timeout S]  (got path {path!r}, timeout {timeout})', err)
        return exit_codes.USAGE
    try:
        base, token = resolve_target()
    except CommandError as error:
        say('backend_probe', str(error), err)
        return error.code
    try:
        code = asyncio.run(_probe_once(base, token, path, timeout, transport))
    except Exception as error:  # noqa: BLE001 - an unusable URL is a probe that got no answer
        say('backend_probe', f'unexpected error: {error!r}', err)
        code = '000'
    print(code, file=stdout if stdout is not None else sys.stdout)
    if is_alive(code):
        return exit_codes.OK
    say('backend_probe', f'GET {base}{path} answered {code}; only a 2xx on that path means the backend is serving it', err)
    return exit_codes.TRANSPORT


__all__ = ['classify_identity', 'fetch_identity', 'is_alive', 'probe', 'run_backend_identity', 'run_backend_probe']
