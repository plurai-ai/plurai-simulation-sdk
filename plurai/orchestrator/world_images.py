"""Images in the finalize world become storage objects: ``image`` is a reference, never the bytes.

The child's finalize world names each produced image by its file (``_image_path``, with ``image_media_type`` and
the file's ``sha256``; see ``plurai.file_rows``), so no image bytes cross the child's pipe. :func:`upload_world_images`
uploads each file to the backend storage with the row's ``sha256`` as the checksum, which storage verifies when the
upload completes, so a file that changed since the snapshot is refused rather than stored; it rewrites the row
to ``image: <scheme>:///storage/leases/<lease_id>`` with ``image_media_type`` and ``image_sha256``, so the
judge side resolves the reference and both carriers (the request and the transcript) hold the same small row.

``_image_path`` never outlives this call, whatever happens. An image that cannot be stored never fails the
scenario: its row loses ``_image_path`` and says why in ``_unreadable['image']``, and the world goes on without it.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from plurai.file_rows import IMAGE_KEY, IMAGE_MEDIA_TYPE_KEY, IMAGE_PATH_KEY, IMAGE_SUFFIXES, UNREADABLE_KEY

if TYPE_CHECKING:
    from plurai.ingest.storage_client import StorageClient

logger = logging.getLogger(__name__)

#: Builds the storage client that takes the uploads.
type StorageClientFactory = Callable[[], StorageClient]

#: The row key with the sha256 (hex) of the stored image's bytes.
IMAGE_SHA256_KEY = 'image_sha256'

#: The reference a stored image is known by on the judge side: the id of a lease on its storage object, the
#: same handle every other SDK upload is resolved by.
STORAGE_REFERENCE_PREFIX = 'plurai:///storage/leases/'

#: A storage object name is at most 64 characters (``MAX_OBJECT_NAME_LENGTH``): half of a sha256 hex
#: digest names the bytes uniquely enough and leaves room for the extension.
_NAME_DIGEST_CHARS = 32

_SHA256 = re.compile(r'[0-9a-f]{64}')

#: The file extension each media type is stored under.
_EXTENSIONS = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/gif': 'gif', 'image/webp': 'webp'}


class _BadImageError(Exception):
    """An image row that cannot be uploaded for a reason of its own (no such file, not an image file, a file
    that changed since the snapshot): the next image is still tried."""


@dataclass(frozen=True, slots=True)
class _Verified:
    """An image row that names a file this process may upload: its path, size, checksum and media type."""

    path: Path
    size: int
    digest: str
    media_type: str


async def upload_world_images(world: dict[str, Any] | None, *, client_factory: StorageClientFactory) -> dict[str, Any] | None:
    """``world`` with every ``_image_path`` row uploaded and rewritten to its storage reference; the input is not mutated.

    ``world`` is the child's finalize world, ``{session_id, state: {entity_id: row}}``. A world with no ``_image_path`` is returned as it is and no
    client is built. A file is uploaded only if its row names it by an absolute path to a regular file (not a link), with an
    image suffix, a supported ``image_media_type`` and a ``sha256``. Identical bytes are uploaded once. A refusal of one upload (a 4xx)
    or a bad file costs that row its image; any other failure (the service unreachable, no bearer
    configured) costs every image not yet uploaded theirs, without trying each one. One warning names the
    distinct causes.
    """
    if world is None:
        return None
    entities = world.get('state')
    if not isinstance(entities, Mapping):
        return world
    pending = {entity_id: row for entity_id, row in entities.items() if isinstance(row, Mapping) and IMAGE_PATH_KEY in row}
    if not pending:
        return world

    stored: dict[str, str] = {}  # sha256 -> the reference to the uploaded bytes
    verified: dict[str, _Verified] = {}  # entity id -> its checked file
    failures: dict[str, str] = {}  # entity id -> the reason it has no image
    causes: dict[str, int] = {}
    aborted: str | None = None
    client: StorageClient | None = None
    try:
        for entity_id, row in pending.items():
            if aborted is not None:
                failures[entity_id] = aborted
                continue
            try:
                image = verified[entity_id] = _verify(row)
                if image.digest not in stored:
                    if client is None:
                        client = client_factory()
                    stored[image.digest] = await _upload(client, image)
            except Exception as exc:  # an image that cannot be stored never fails the scenario
                reason = str(exc) if isinstance(exc, _BadImageError) else f'upload failed: {_describe(exc)}'
                failures[entity_id] = reason
                causes[reason] = causes.get(reason, 0) + 1
                if not _is_per_image(exc):
                    aborted = reason
    finally:
        if client is not None:
            await client.aclose()
    if causes:
        logger.warning(
            'the finalize world has %d image(s) that could not be stored and are sent without the image (%s)',
            len(failures),
            '; '.join(f'{reason} x{count}' for reason, count in causes.items()),
        )

    replaced: dict[str, Any] = {}
    for entity_id, row in pending.items():
        base = {key: value for key, value in row.items() if key != IMAGE_PATH_KEY}
        if entity_id in failures:
            notes = base.get(UNREADABLE_KEY)
            replaced[entity_id] = {
                **{key: value for key, value in base.items() if key not in {IMAGE_KEY, IMAGE_MEDIA_TYPE_KEY}},
                UNREADABLE_KEY: {**(notes if isinstance(notes, Mapping) else {}), IMAGE_KEY: failures[entity_id]},
            }
        else:
            image = verified[entity_id]
            replaced[entity_id] = {**base, IMAGE_KEY: stored[image.digest], IMAGE_MEDIA_TYPE_KEY: image.media_type, IMAGE_SHA256_KEY: image.digest}
    return {**world, 'state': {**entities, **replaced}}


def _verify(row: Mapping[str, Any]) -> _Verified:
    """The row's file, checked without reading it: an absolute path to a regular file at the very path it names
    (no link), an image suffix, a supported media type and a sha256. The size is the file's now and the checksum
    the row's, so storage, which hashes what it receives, refuses a file that changed since the snapshot."""
    named = row.get(IMAGE_PATH_KEY)
    if not isinstance(named, str) or not named:
        raise _BadImageError('the row names no image file')
    path = Path(named)
    if not path.is_absolute():
        raise _BadImageError('the image path is not absolute')
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise _BadImageError(f'{path.suffix or "the file"} is not an image suffix')
    media_type = row.get(IMAGE_MEDIA_TYPE_KEY)
    if media_type not in _EXTENSIONS:
        raise _BadImageError(f'{media_type!r} is not a supported image media type')
    digest = row.get('sha256')
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise _BadImageError('the row has no sha256 to check the file against')
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise _BadImageError(f'the image file cannot be read: {_describe(exc)}') from exc
    if not stat.S_ISREG(info.st_mode) or os.path.realpath(path) != named:
        raise _BadImageError('the image path is not a regular file (a link, or reached through one)')
    return _Verified(path, info.st_size, digest, media_type)


def storage_client_for(base_url: str, token: str | None) -> StorageClientFactory:
    """A factory for the storage client of ONE backend: the run's own base URL and bearer, which are not
    always what the environment names."""

    def build() -> StorageClient:
        from plurai.ingest.storage_client import StorageClient

        return StorageClient(base_url, token)

    return build


def _describe(exc: Exception) -> str:
    """``exc`` as ``Class: message`` for a row and a log line. An ``OSError`` says only what the system said
    (``strerror``): its own message names the file, and a local path never outlives the call."""
    return f'{type(exc).__name__}: {exc.strerror or type(exc).__name__}' if isinstance(exc, OSError) else f'{type(exc).__name__}: {exc}'


def _is_per_image(exc: Exception) -> bool:
    """Whether the failure is this image's alone (a bad file, a refused upload), so the next one is still tried.

    A credit refusal is the organization's, so it ends the uploads.
    """
    from plurai.ingest.storage_client import StorageCreditStopError, StorageUploadError

    return isinstance(exc, _BadImageError | StorageUploadError) and not isinstance(exc, StorageCreditStopError)


async def _upload(client: StorageClient, image: _Verified) -> str:
    name = f'{image.digest[:_NAME_DIGEST_CHARS]}.{_EXTENSIONS[image.media_type]}'
    object_id, limits = await client.create_storage_object(name=name, content_type=image.media_type, content_length=image.size, checksum=image.digest)
    await client.upload_content(object_id, image.path, image.size, limits)
    lease_id = await client.create_storage_lease(object_id)
    return f'{STORAGE_REFERENCE_PREFIX}{lease_id}'
