"""Wire models for the backend ``/storage`` service.

These mirror the storage subset of the backend's OpenAPI-generated client models
(``StorageObject`` / ``Metadata`` / ``Metadata2`` / ``DetailedStorageObjectLease`` /
``StorageObjectsPostRequest`` / ``CreatedStorageObject``) so the SDK client and the
``/storage`` routes bind to ONE contract shape: camelCase wire aliases,
``extra='allow'`` (forward-compatible), ``contentLength > 0`` on the request metadata.

They are **mirrored, not code-generated**, from the backend's storage models.

:class:`PartUploadLimits` mirrors an addition the backend makes to that contract: the create
response's ``partUpload``, which tells a client the part size and count it uploads by. It is
optional here so that a backend which omits it, and so cannot take the parts-only upload, is
reported as that rather than as an unreadable answer.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel

_MAX_INT = 9007199254740991  # 2**53 - 1, the JSON-safe integer ceiling in the contract


class Metadata(BaseModel):
    """Response metadata of a storage object."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    content_type: Annotated[str, Field(alias='contentType', min_length=1)]
    content_length: Annotated[int, Field(alias='contentLength', gt=0, le=_MAX_INT)]
    checksum: Annotated[str, Field(min_length=1)]
    extra: dict[str, Any] | None = {}


class Metadata2(BaseModel):
    """Request metadata for creating a storage object (``contentLength > 0``)."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    content_type: Annotated[str, Field(alias='contentType', min_length=1)]
    content_length: Annotated[int, Field(alias='contentLength', gt=0, le=_MAX_INT)]
    checksum: Annotated[str, Field(min_length=1)]
    extra: dict[str, Any] | None = {}


class StorageObject(BaseModel):
    """A storage object stored in the storage system."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    id: UUID
    name: str
    provider_name: Annotated[str, Field(alias='providerName')]
    metadata: Metadata


class StorageObjectsPostRequest(BaseModel):
    """``POST /storage/objects`` request body."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    name: Annotated[str, Field(min_length=1, max_length=64)]
    prefix: str | None = None
    storage_provider_name: Annotated[str | None, Field(alias='storageProviderName', min_length=1)] = None
    metadata: Metadata2


class PartUploadLimits(BaseModel):
    """How the backend takes content in parts: at most ``max_part_bytes`` per part, ``max_parts`` parts."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    max_part_bytes: Annotated[int, Field(alias='maxPartBytes', ge=1)]
    max_parts: Annotated[int, Field(alias='maxParts', ge=1)]


class NewStorageObject(StorageObject):
    """The storage object a create returns, with the part-upload limits."""

    part_upload: Annotated[PartUploadLimits | None, Field(alias='partUpload')] = None


class CreatedStorageObject(RootModel[NewStorageObject]):
    """``POST /storage/objects`` response (a bare storage object)."""

    root: NewStorageObject


class DetailedStorageObjectLease(BaseModel):
    """A lease (post-upload READ handle) plus its associated storage object."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    id: UUID
    storage_object_id: Annotated[UUID, Field(alias='storageObjectId')]
    expires_at: Annotated[AwareDatetime | None, Field(alias='expiresAt')] = None
    storage_object: Annotated[StorageObject, Field(alias='storageObject')]


__all__ = [
    'CreatedStorageObject',
    'DetailedStorageObjectLease',
    'Metadata',
    'Metadata2',
    'NewStorageObject',
    'PartUploadLimits',
    'StorageObject',
    'StorageObjectsPostRequest',
]
