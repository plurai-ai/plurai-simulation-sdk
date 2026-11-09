"""What every ``plurai`` command that talks to the backend over plain HTTP shares.

A command resolves its target from the configuration (:func:`resolve_target`), sends the bearer only
when one is set (:func:`client`), classifies each answer into an exit code (:class:`CommandError`),
keeps stdout for the one value a caller parses and puts everything else on stderr under its own
prefix (:func:`say`). Files land through a temp file and a rename (:func:`write_atomic`), so a
failed command never truncates a good one. The exit codes below are the rows the commands have in
common; each command's docstring lists the rows it uses, and :mod:`plurai.exit_codes` holds the table.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import TextIO

import httpx

from plurai import _http, client_handshake, exit_codes
from plurai._http_retry import RATE_LIMIT_RETRIES, RATE_LIMITED_STATUS, rate_limit_wait
from plurai.credit import credit_refusal_sentence
from plurai.customer_safe import describe_failure
from plurai.ingest.storage_client import StorageConfigError, resolve_configured_target
from plurai.orchestrator._retry import TRANSIENT_STATUS_CODES, TRANSIENT_TRANSPORT_ERRORS

#: An edge that upgrades http to https answers one redirect; a chain longer than this is a loop or a login flow.
MAX_REDIRECTS = 3


class CommandError(Exception):
    """A classified failure: ``code`` is the exit code, the message is the stderr line."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def upgrade_required(response: httpx.Response) -> CommandError:
    """The classified failure for a 426: the ``pip`` line that upgrades the SDK, under the URL that refused it."""
    return CommandError(exit_codes.UPGRADE_REQUIRED, f'{response.url}: {client_handshake.upgrade_message_for(response)}')


def credit_stop(response: httpx.Response) -> CommandError | None:
    """The classified failure for a 402 whose body names a credit kind, else ``None``.

    The gateway in front of the backend answers every route with that 402 at an organization's hard stop. It is the
    credit stop whatever the command: ``JOB_FAILED``, the code the job commands use for it, and the kind's sentence, never the
    status or the body.
    """
    sentence = credit_refusal_sentence(response)
    return None if sentence is None else CommandError(exit_codes.JOB_FAILED, sentence)


def say(prefix: str, message: str, err: TextIO) -> None:
    print(f'{prefix}: {message}', file=err)


def resolve_target() -> tuple[str, str | None]:
    """Base and bearer from the configuration, ``PLURAI_URL`` with ``PLURAI_TOKEN`` or else the saved login
    (:func:`~plurai.ingest.storage_client.resolve_configured_target`), never from an argument.

    The base is parsed by httpx itself, so a port that is not a number or a stray control
    character is refused here, before any request. Userinfo is refused because httpx turns it
    into Basic auth in place of the bearer.

    Raises:
        CommandError: ``CREDENTIALS`` when nothing is configured, ``REFUSED`` for an unusable target.
    """
    try:
        base, token = resolve_configured_target()
    except StorageConfigError as error:
        raise CommandError(exit_codes.CREDENTIALS, f'ERROR: {error}') from error
    except ValueError as error:
        raise CommandError(exit_codes.REFUSED, f'ERROR: unusable backend URL ({error})') from error
    try:
        url = httpx.URL(base)
    except httpx.InvalidURL as error:
        raise CommandError(exit_codes.REFUSED, f'ERROR: unusable backend URL {base!r}: {error}') from error
    if url.scheme not in ('http', 'https') or not url.host or url.userinfo:
        raise CommandError(exit_codes.REFUSED, f'ERROR: unusable backend URL {base!r}: expected http(s)://host[:port][/prefix]')
    return base, token


def client(
    base: str, token: str | None, timeout: httpx.Timeout | float, transport: httpx.AsyncBaseTransport | None = None, *, follow_redirects: bool = False
) -> httpx.AsyncClient:
    """A client for ``base`` that sends the bearer only when one is set: loopback dev is
    legitimately tokenless, and an empty ``Bearer`` header is not. ``transport`` is the test seam.
    httpx drops the bearer when a followed redirect leaves the origin."""
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    return _http.AsyncClient(
        base_url=base, headers=headers, timeout=timeout, transport=transport, follow_redirects=follow_redirects, max_redirects=MAX_REDIRECTS
    )


async def send_with_retry(
    send: Callable[[], Awaitable[httpx.Response]], *, attempts: int, delay: float, timeout: float, retry_statuses: frozenset[int] = frozenset()
) -> httpx.Response:
    """``send()``, retried ``delay`` seconds after a transport failure or a status in
    ``retry_statuses``; each attempt ends within ``timeout``. The last response is returned
    whatever its status, so the caller classifies it. A retry logs nothing: the command's
    classified line is the one diagnostic, under its own prefix.

    A 429 is the backend throttling the caller, so it is always waited out and never counted
    against ``attempts``: the wait follows ``Retry-After`` under the SDK-wide budget
    (:mod:`plurai._http_retry`), whatever ``retry_statuses`` holds. The request was refused before
    it was forwarded, so repeating it cannot apply it twice. A throttle that outlasts the budget
    is returned as the last response.

    Raises:
        httpx.HTTPError | TimeoutError: the last transport failure, once the attempts are spent.
    """
    attempt = 0
    throttled = 0
    while True:
        attempt += 1
        try:
            response = await asyncio.wait_for(send(), timeout)
        except (*TRANSIENT_TRANSPORT_ERRORS, TimeoutError):
            if attempt == attempts:
                raise
        else:
            if response.status_code == RATE_LIMITED_STATUS and throttled < RATE_LIMIT_RETRIES:
                throttled += 1
                attempt -= 1
                await asyncio.sleep(rate_limit_wait(throttled, response.headers.get('Retry-After')))
                continue
            if attempt == attempts or response.status_code not in retry_statuses:
                return response
        await asyncio.sleep(delay)


async def get_with_retry(
    http: httpx.AsyncClient, path: str, *, attempts: int, delay: float, timeout: float, retry_statuses: frozenset[int] = frozenset()
) -> httpx.Response:
    """``GET path`` through :func:`send_with_retry`."""
    return await send_with_retry(lambda: http.get(path), attempts=attempts, delay=delay, timeout=timeout, retry_statuses=retry_statuses)


@contextlib.contextmanager
def prefixed_warnings(prefix: str, err: TextIO) -> Iterator[None]:
    """Inside the block, a warning from any ``plurai`` logger reaches ``err`` as ``<prefix>: <message>``.

    With no handler configured, Python's last-resort handler would print it bare on stderr (a
    retry inside :class:`~plurai.ingest.storage_client.StorageClient`, say), and the plugin reads
    every stderr line of a command by its prefix. The records do not propagate inside the block,
    so a handler the host configured on the root logger does not print them a second time.
    """
    logger = logging.getLogger('plurai')
    handler = logging.StreamHandler(err)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter(f'{prefix}: %(message)s'))
    propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate


def write_atomic(dest: Path, content: bytes) -> None:
    """``content`` to ``dest`` through a temp file beside it and a rename; on failure the temp
    file is removed and ``dest`` is untouched.

    Raises:
        OSError: from the write or the rename.
    """
    tmp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
    try:
        tmp.write_bytes(content)
        tmp.replace(dest)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


__all__ = [
    'MAX_REDIRECTS',
    'TRANSIENT_STATUS_CODES',
    'CommandError',
    'client',
    'describe_failure',
    'get_with_retry',
    'prefixed_warnings',
    'resolve_target',
    'say',
    'send_with_retry',
    'write_atomic',
]
