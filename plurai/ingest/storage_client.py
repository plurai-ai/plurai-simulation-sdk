"""httpx client for the backend ``/storage`` service.

Sits beside :class:`plurai.orchestrator.client.OrchestratorClient`. The storage service is served
by the backend at ``<base>/storage/*`` (objects + leases), behind the same bearer the MCP client
and telemetry use, not a separate external object store.

* Auth is the Bearer ``Authorization: Bearer <token>`` of the resolved endpoint; no token →
  NO ``Authorization`` header, which only a loopback base (tokenless dev) can resolve to; a
  non-loopback base with no token is a configuration error at resolution (fail-closed — the
  deployed box requires the bearer).
* The base URL and token resolve through :func:`plurai._endpoints.resolve` (the ``PLURAI_URL`` /
  ``PLURAI_TOKEN`` pair, else the saved login); there is no default. The client targets
  ``<base>/storage/objects`` etc.; the resolved base is readable back via
  :attr:`StorageClient.base_url` (the CLI's disclosure/audit target).
* Content goes up one way, whatever its size: ``POST /storage/objects`` (which answers with the
  part limits under ``partUpload``), raw ``application/octet-stream`` bodies to
  ``PUT /storage/objects/{id}/parts/{i}``, then ``POST /storage/objects/{id}/complete``. A small
  object is one part. A part is the smaller of the advertised size and :data:`MAX_PART_BYTES`, so
  every request body fits a relay that bounds it (1 MiB): the relay's limit is the client's to respect, and a backend configured for larger
  parts still gets parts that cross it. A create response without ``partUpload`` comes from a
  backend that cannot take this upload, and raises :class:`PartUploadUnsupportedError`.
* Transient retries ({502,503,504} + connect, full-jitter exponential backoff
  0.5→2.0s) reuse the risk-scoped policy in :mod:`plurai.orchestrator._retry`. A part PUT and the
  completion are idempotent, so a failed part is re-sent alone; the create is not, since each one
  mints an object. A 4xx is a refusal no retry can fix, and raises :class:`StorageUploadError`
  with the backend's reason.
* A **lease is a post-upload READ handle** — its ``id`` is the reference the caller
  passes to ``finalize`` (:meth:`upload_file` returns it).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from plurai import _http, client_handshake
from plurai._endpoints import EndpointError, resolve
from plurai.credit import CREDIT_REFUSED_STATUS, credit_sentence
from plurai.customer_safe import describe_failure
from plurai.host_effects import register_backend_token
from plurai.ingest.storage_models import (
    CreatedStorageObject,
    DetailedStorageObjectLease,
    Metadata2,
    PartUploadLimits,
    StorageObject,
    StorageObjectsPostRequest,
)
from plurai._http_retry import RATE_LIMITED_STATUS
from plurai.login._environments import billing_url_for
from plurai.network_refusal import NetworkRefusal, note, sentence, undelivered
from plurai.orchestrator._retry import TRANSIENT_TRANSPORT_ERRORS, CreditRefused, NetworkRefused, RetryConfig, request_with_retry

logger = logging.getLogger(__name__)

#: Storage-object name limit + default request timeout (seconds).
MAX_OBJECT_NAME_LENGTH = 64
DEFAULT_TIMEOUT = 30.0

#: The largest part this client sends, whatever the backend advertises. Base64 inside a 1 MiB
#: JSON envelope makes 512 KiB about 683 KiB, leaving room for the envelope.
MAX_PART_BYTES = 512 * 1024

#: The slowest rate ``complete`` is budgeted for. The backend reads, hashes and writes every part
#: before it answers, so a fixed timeout that fits a small object times out a large one while the
#: server is still working on it.
COMPLETE_MIN_BYTES_PER_SECOND = 1024 * 1024

#: Storage retry policy: full-jitter exponential backoff 0.5s → 2.0s, 2 extra attempts.
_STORAGE_RETRY = RetryConfig(max_retries=2, backoff_base=0.5, backoff_cap=2.0, quiet_transient=False)

#: The most of a refusal's body quoted in :class:`StorageUploadError`, so an HTML error page
#: from a front cannot flood the one-line CLI error.
_MAX_REFUSAL_DETAIL_CHARS = 300


class StorageConfigError(ValueError):
    """Raised on invalid storage configuration: no usable backend configuration, an
    empty explicitly-passed base URL, or a bad storage object name."""


class PartUploadUnsupportedError(Exception):
    """The create response carries no ``partUpload``: the backend does not support the parts-only upload,
    so no request this client sends can deliver the content. Upgrading the backend fixes it; a
    retry cannot. The object the create minted stays behind empty."""


class StorageUploadError(Exception):
    """The backend refused a storage request with a 4xx: a checksum or length mismatch, a part
    over the size limit, missing parts, an upload already complete, a rejected bearer.

    A retry cannot fix any of these, so the CLI reports it as one line and exits with its
    fail-closed code. ``status_code`` and ``endpoint`` (the route template) are kept for callers
    that branch on them; the message quotes the backend's own reason.
    """

    def __init__(self, message: str, *, status_code: int, endpoint: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


class StorageCreditStopError(StorageUploadError):
    """The gateway refused a storage request for credit: a 402 whose body names a credit kind.

    The message is the kind's sentence, never the gateway's status or text, so any caller that prints the error
    prints exactly the credit stop. A command maps it to its credit exit code rather than to a storage refusal.
    """


class StorageNetworkRefusedError(StorageUploadError):
    """A storage request refused by something other than Plurai (:class:`~plurai.orchestrator._retry.NetworkRefused`):
    a 4xx a proxy or firewall wrote. The message is :func:`plurai.network_refusal.sentence`; ``refusal`` is the record.
    """

    def __init__(self, message: str, *, status_code: int, endpoint: str, refusal: NetworkRefusal) -> None:
        super().__init__(message, status_code=status_code, endpoint=endpoint)
        self.refusal = refusal


class StorageContentUndelivered(httpx.NetworkError):  # noqa: N818  # an httpx transport error, named for what the operator reads
    """A part of an object the backend had already created failed to arrive: the transport failure the retries ended on.

    An :exc:`httpx.NetworkError`, so every handler of the transport failure still catches it. It is not an
    :exc:`httpx.TimeoutException`, even when the failure was a timeout; the original is its ``__cause__``. The message is
    :func:`plurai.network_refusal.sentence`; ``refusal`` is the record.
    """

    def __init__(self, refusal: NetworkRefusal, *, request: httpx.Request | None) -> None:
        super().__init__(sentence(refusal), request=request)
        self.refusal = refusal


#: The route template of a part PUT.
_PART_ENDPOINT = '/storage/objects/{id}/parts/{index}'


def _request_of(error: BaseException) -> httpx.Request | None:
    try:
        request = getattr(error, 'request', None)
    except RuntimeError:  # an httpx error raised without a request
        return None
    return request if isinstance(request, httpx.Request) else None


def _refusal_detail(response: httpx.Response) -> str:
    """The backend's reason for a refusal: the scrubbed ``error``/``message``/``detail`` field of a JSON body, cut to
    :data:`_MAX_REFUSAL_DETAIL_CHARS`, with the error id to quote. A body that is not JSON contributes nothing."""
    return describe_failure(response, limit=_MAX_REFUSAL_DETAIL_CHARS)


@dataclass(frozen=True)
class UploadResult:
    """Outcome of :meth:`StorageClient.upload_file` — the lease ref + audit metadata.

    ``lease_id`` is the primary reference handed to ``finalize``; ``checksum`` and
    ``bytes`` feed the lease-ref manifest + audit log without re-reading the file.
    """

    lease_id: str
    object_id: str
    checksum: str
    bytes: int


@dataclass(frozen=True)
class PartUpload:
    """Outcome of :meth:`StorageClient.upload_if_absent`: the object that holds the bytes, and whether
    this call had to send them (``False`` when the backend already held exactly those bytes)."""

    object_id: str
    uploaded: bool


def _validate_object_name(name: str) -> None:
    """Mirror the backend's storage-object name rule (1..64 chars, no path separators)."""
    if not name:
        raise StorageConfigError('storage object name cannot be empty')
    if len(name) > MAX_OBJECT_NAME_LENGTH:
        raise StorageConfigError(f'storage object name too long: {len(name)} chars (max {MAX_OBJECT_NAME_LENGTH})')
    if '/' in name or '\\' in name:
        raise StorageConfigError(f'storage object name must not contain path separators: {name!r}')


def resolve_configured_target() -> tuple[str, str | None]:
    """The base URL and bearer the configuration names (:func:`plurai._endpoints.resolve`: the
    ``PLURAI_URL`` / ``PLURAI_TOKEN`` pair, else the saved login), for a caller that also sends
    non-storage requests to the same backend. The token is registered for redaction.

    Raises:
        StorageConfigError: Nothing is configured, the configuration is partial, or the saved login is unusable.
    """
    try:
        endpoint = resolve()
    except EndpointError as error:
        raise StorageConfigError(str(error)) from error
    register_backend_token(endpoint.token)
    return endpoint.url, endpoint.token


class StorageClient:
    """Async httpx client for the backend ``/storage`` contract (8 routes)."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        retry: RetryConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_part_bytes: int = MAX_PART_BYTES,
    ) -> None:
        if not base_url:
            raise StorageConfigError('storage base URL is required')
        self._base_url = base_url.rstrip('/')
        # Bearer auth mirrors telemetry + the MCP client: the backend token when set,
        # NO Authorization header when unset (unauthenticated dev mode). Only Authorization
        # is a default header: the JSON POSTs and the raw part PUTs each set their own Content-Type.
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        self._client = _http.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,  # DI seam for the SDK-local contract test double
        )
        self._timeout = timeout
        self._retry = retry if retry is not None else _STORAGE_RETRY
        self._max_part_bytes = max_part_bytes

    @property
    def base_url(self) -> str:
        """The resolved storage base — the EXACT base every request targets. The CLI reads
        its disclosure/audit target off this property (not off the env), so what is
        disclosed to the human and where the bytes actually go cannot drift."""
        return self._base_url

    @classmethod
    def from_env(cls, *, timeout: float = DEFAULT_TIMEOUT, transport: httpx.AsyncBaseTransport | None = None) -> StorageClient:
        """Build a client for the configured backend — one base URL + its bearer
        (:func:`resolve_configured_target`).

        ONE server serves every surface — ``/storage/*`` rides the same base as REST and ``/mcp``.
        A non-loopback base with no token cannot resolve (fail-closed), so an unauthenticated
        remote call is never built.
        """
        base, token = resolve_configured_target()
        return cls(base, token, timeout=timeout, transport=transport)

    async def __aenter__(self) -> StorageClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, call: Any, *, idempotent: bool, endpoint: str) -> httpx.Response:  # noqa: ANN401 - awaitable-returning thunk
        """Send with the storage retry policy; a 4xx raises :class:`StorageUploadError`, a credit 402 its subclass :class:`StorageCreditStopError`,
        and a 4xx that did not come from Plurai its subclass :class:`StorageNetworkRefusedError`.

        A 5xx that outlasts the retries still raises :class:`httpx.HTTPStatusError`: that is a
        server fault, not a refusal of this upload. So does a 429: the backend is throttling the
        caller, which says nothing about the upload, and a later try can succeed.
        """
        try:
            return await request_with_retry(call, config=self._retry, idempotent=idempotent, endpoint=endpoint)
        except CreditRefused as exc:
            billing_url = billing_url_for(str(exc.response.url)) if exc.response is not None else ''
            sentence = credit_sentence(exc.credit.code, resets_at=exc.credit.resets_at, billing_url=billing_url)
            raise StorageCreditStopError(sentence, status_code=CREDIT_REFUSED_STATUS, endpoint=endpoint) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if not 400 <= status < 500 or status == RATE_LIMITED_STATUS:
                raise
            if isinstance(exc, NetworkRefused):
                raise StorageNetworkRefusedError(str(exc), status_code=status, endpoint=endpoint, refusal=exc.refusal) from exc
            raise StorageUploadError(
                client_handshake.upgrade_message_for(exc.response)
                if status == client_handshake.UPGRADE_STATUS
                else f'storage refused {exc.request.method} {endpoint} with {status}: {_refusal_detail(exc.response)}',
                status_code=status,
                endpoint=endpoint,
            ) from exc

    async def create_storage_object(
        self,
        name: str,
        content_type: str,
        content_length: int,
        checksum: str,
        *,
        storage_provider_name: str | None = None,
        prefix: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, PartUploadLimits]:
        """``POST /storage/objects`` (non-idempotent) → the new object's id and the part-upload
        limits the backend advertises.

        Raises:
            PartUploadUnsupportedError: The response carries no ``partUpload``.
        """
        _validate_object_name(name)
        body = StorageObjectsPostRequest(
            name=name,
            prefix=prefix,
            storage_provider_name=storage_provider_name,
            metadata=Metadata2(
                content_type=content_type,
                content_length=content_length,
                checksum=checksum,
                extra=extra_metadata or {},
            ),
        ).model_dump(by_alias=True, exclude_none=True)
        response = await self._request(
            lambda: self._client.post('/storage/objects', json=body),
            idempotent=False,
            endpoint='/storage/objects',
        )
        created = CreatedStorageObject.model_validate(response.json()).root
        if created.part_upload is None:
            raise PartUploadUnsupportedError(
                f'storage backend {self._base_url} predates part uploads (no partUpload on the POST /storage/objects answer), '
                'so it cannot take any upload from this SDK; upgrade the backend'
            )
        return str(created.id), created.part_upload

    async def upload_content(self, object_id: str, source: bytes | Path, content_length: int, limits: PartUploadLimits) -> None:
        """Send ``content_length`` bytes of ``source`` as parts of at most the part size, then complete.
        ``limits`` is what :meth:`create_storage_object` returned for the object.

        Each part is a raw octet-stream body read at its offset on every attempt, so a large file is
        never held whole and a retried part is re-read rather than resent from a consumed stream.
        Part PUTs and the completion are idempotent on the server (a re-sent index replaces the
        part; completing a complete object returns it), so both retry, each on its own.
        """
        part_bytes = min(limits.max_part_bytes, self._max_part_bytes)
        # limits.max_parts is not checked: at 512 KiB parts the backend's budget (10_000 parts, 5_000 MiB)
        # covers its per-object cap (128 MiB, 256 parts), and it refuses an object over that cap at
        # create with a 413, before any part is sent.
        part_count = math.ceil(content_length / part_bytes)
        for index in range(part_count):
            offset = index * part_bytes
            size = min(part_bytes, content_length - offset)
            url = f'/storage/objects/{object_id}/parts/{index}'

            async def _put_part_once(url: str = url, offset: int = offset, size: int = size) -> httpx.Response:
                if isinstance(source, (bytes, bytearray)):
                    body = bytes(source[offset : offset + size])
                else:
                    # Blocking read of one part is intentional: this is a one-shot CLI uploader, not a
                    # server event loop.
                    with Path(source).open('rb') as handle:  # noqa: ASYNC230
                        handle.seek(offset)
                        body = handle.read(size)
                return await self._client.put(url, content=body, headers={'Content-Type': 'application/octet-stream'})

            try:
                await self._request(_put_part_once, idempotent=True, endpoint=_PART_ENDPOINT)
            except TRANSIENT_TRANSPORT_ERRORS as exc:
                request = _request_of(exc)
                refusal = undelivered(exc, request=request, request_kind=_PART_ENDPOINT)
                note(refusal)
                raise StorageContentUndelivered(refusal, request=request) from exc
        complete_timeout = self._complete_timeout(content_length)
        await self._request(
            lambda: self._client.post(f'/storage/objects/{object_id}/complete', json={'partCount': part_count}, timeout=complete_timeout),
            idempotent=True,
            endpoint='/storage/objects/{id}/complete',
        )

    def _complete_timeout(self, content_length: int) -> float:
        """The ``complete`` timeout: the client timeout, or longer for an object big enough that
        reading it at :data:`COMPLETE_MIN_BYTES_PER_SECOND` takes longer."""
        return max(self._timeout, content_length / COMPLETE_MIN_BYTES_PER_SECOND)

    async def create_storage_lease(self, object_id: str) -> str:
        """``POST /storage/objects/{id}/lease`` → the new lease id (the finalize ref)."""
        response = await self._request(
            lambda: self._client.post(f'/storage/objects/{object_id}/lease', json={}),
            idempotent=False,
            endpoint='/storage/objects/{id}/lease',
        )
        return str(DetailedStorageObjectLease.model_validate(response.json()).id)

    async def find_content(self, checksum: str, content_length: int) -> str | None:
        """``GET /storage/content`` → the id of a completed object holding exactly these bytes, or ``None``.

        The answer is about a hash and a length only: no content is sent. Any 4xx raises
        :class:`StorageUploadError`. A 404 means the backend predates content-addressed parts: a part it cannot
        look up by hash is a part it cannot ground, so the caller must not send the bytes.
        """
        try:
            response = await self._request(
                lambda: self._client.get('/storage/content', params={'checksum': checksum, 'contentLength': content_length}),
                idempotent=True,
                endpoint='/storage/content',
            )
        except StorageUploadError as refused:
            if refused.status_code == 404:
                raise StorageUploadError(
                    'the backend has no GET /storage/content route: it predates content-addressed parts and cannot ground what is sent to it. '
                    'Nothing was uploaded; use a backend that serves the route.',
                    status_code=404,
                    endpoint='/storage/content',
                ) from refused
            raise
        body = response.json()
        object_id = body.get('objectId') if isinstance(body, dict) and body.get('found') else None
        return str(object_id) if object_id else None

    async def upload_if_absent(
        self,
        path: str | Path,
        *,
        name: str,
        checksum: str,
        content_length: int,
        content_type: str = 'application/gzip',
        storage_provider_name: str | None = None,
    ) -> PartUpload:
        """Ask the backend for these bytes by hash; send them (object → parts → complete) only when it lacks them.

        No lease is made: a grounding part is referenced by its hash, and the backend pins the object itself.
        """
        held = await self.find_content(checksum, content_length)
        if held is not None:
            return PartUpload(object_id=held, uploaded=False)
        object_id, limits = await self.create_storage_object(
            name=name,
            content_type=content_type,
            content_length=content_length,
            checksum=checksum,
            storage_provider_name=storage_provider_name,
        )
        await self.upload_content(object_id, Path(path), content_length, limits)
        return PartUpload(object_id=object_id, uploaded=True)

    async def get_storage_object(self, object_id: str) -> StorageObject:
        """``GET /storage/objects/{id}`` → the storage object."""
        response = await self._request(
            lambda: self._client.get(f'/storage/objects/{object_id}'),
            idempotent=True,
            endpoint='/storage/objects/{id}',
        )
        return StorageObject.model_validate(response.json())

    async def get_storage_lease(self, lease_id: str) -> DetailedStorageObjectLease:
        """``GET /storage/leases/{id}`` → the detailed lease."""
        response = await self._request(
            lambda: self._client.get(f'/storage/leases/{lease_id}'),
            idempotent=True,
            endpoint='/storage/leases/{id}',
        )
        return DetailedStorageObjectLease.model_validate(response.json())

    async def download_storage_lease_content(self, lease_id: str) -> bytes:
        """``GET /storage/leases/{id}/content`` → raw bytes (build-job read path)."""
        response = await self._request(
            lambda: self._client.get(f'/storage/leases/{lease_id}/content'),
            idempotent=True,
            endpoint='/storage/leases/{id}/content',
        )
        return response.content

    async def download_storage_object_content(self, object_id: str) -> bytes:
        """``GET /storage/objects/{id}/content`` → raw bytes."""
        response = await self._request(
            lambda: self._client.get(f'/storage/objects/{object_id}/content'),
            idempotent=True,
            endpoint='/storage/objects/{id}/content',
        )
        return response.content

    async def upload_file(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        content_type: str = 'application/octet-stream',
        storage_provider_name: str | None = None,
    ) -> UploadResult:
        """sha256 → create object → parts → complete → create lease → return the lease ref.

        Returns an :class:`UploadResult` (lease id + checksum + byte count) so the
        caller can build the lease-ref manifest / audit log without re-reading the file.
        """
        p = Path(path)
        content = p.read_bytes()  # noqa: ASYNC240 - one-shot CLI uploader, blocking read is fine
        checksum = hashlib.sha256(content).hexdigest()
        object_id, limits = await self.create_storage_object(
            name=name or p.name,
            content_type=content_type,
            content_length=len(content),
            checksum=checksum,
            storage_provider_name=storage_provider_name,
        )
        await self.upload_content(object_id, content, len(content), limits)
        lease_id = await self.create_storage_lease(object_id)
        logger.debug('uploaded %s (%d bytes) → object %s → lease %s', p.name, len(content), object_id, lease_id)
        return UploadResult(lease_id=lease_id, object_id=object_id, checksum=checksum, bytes=len(content))

    async def upload_archive(
        self,
        path: str | Path,
        *,
        name: str,
        checksum: str,
        content_length: int,
        content_type: str = 'application/gzip',
        storage_provider_name: str | None = None,
    ) -> UploadResult:
        """Create object → parts read from the file (never buffered whole) → complete → lease, for
        the ONE single-archive ingest upload.

        Unlike :meth:`upload_file` (which ``read_bytes()`` the whole file into memory), the archive
        is sent from the file, one part-sized read per part attempt, so even a large archive never
        fully buffers in the client. The caller passes
        the archive's already-computed SHA-256 (``checksum``) + size (``content_length``) from the
        streamed build, so this makes no extra hash pass. Returns the single server-minted lease ref.
        """
        object_id, limits = await self.create_storage_object(
            name=name,
            content_type=content_type,
            content_length=content_length,
            checksum=checksum,
            storage_provider_name=storage_provider_name,
        )
        await self.upload_content(object_id, Path(path), content_length, limits)
        lease_id = await self.create_storage_lease(object_id)
        logger.debug('uploaded archive %s (%d bytes) → object %s → lease %s', name, content_length, object_id, lease_id)
        return UploadResult(lease_id=lease_id, object_id=object_id, checksum=checksum, bytes=content_length)


__all__ = [
    'COMPLETE_MIN_BYTES_PER_SECOND',
    'DEFAULT_TIMEOUT',
    'MAX_OBJECT_NAME_LENGTH',
    'MAX_PART_BYTES',
    'PartUpload',
    'PartUploadUnsupportedError',
    'StorageClient',
    'StorageConfigError',
    'StorageContentUndelivered',
    'StorageNetworkRefusedError',
    'StorageUploadError',
    'UploadResult',
    'resolve_configured_target',
]
