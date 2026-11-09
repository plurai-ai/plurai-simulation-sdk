"""The backend's total-uncompressed cap on an ingest archive, checked before any byte leaves.

Every ``*_start`` verb on the backend extracts the uploaded ``.tar.gz`` under one decompression-bomb
guard: the running sum of member sizes may not exceed ``max_uncompressed_bytes``. That check runs
only when the verb extracts, which is after the whole archive has been uploaded. The storage
routes cap only the compressed size, and nothing on the wire tells the client the uncompressed
cap. So the client holds the server default here and refuses an over-cap archive locally.

The check is format-neutral. It reads the member keys and sizes that go into the tar, which is
exactly the sum the server adds up (a tar member's size is its file size). The caller supplies the
remedy text, because only the caller knows what narrows its own export.

Resolution order for the cap: a library caller's explicit value, then the
:data:`MAX_UNCOMPRESSED_BYTES_ENV` variable, then :data:`DEFAULT_MAX_UNCOMPRESSED_BYTES`. The
override exists for a server whose operator raised its own cap; it cannot make an upload pass a
server that did not.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

#: The backend default total-uncompressed cap (128 MiB).
DEFAULT_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024

#: Client-side override for a server whose operator changed the cap above.
MAX_UNCOMPRESSED_BYTES_ENV = 'PLURAI_INGEST_MAX_UNCOMPRESSED_BYTES'

#: How many of the largest members the refusal names.
LARGEST_MEMBERS_SHOWN = 5


class ArchiveCapConfigError(ValueError):
    """The cap override is not a positive integer."""


class ArchiveOverCapError(Exception):
    """The archive would expand past the server's total-uncompressed cap; nothing was uploaded."""


def resolve_max_uncompressed_bytes(explicit: int | None = None) -> int:
    """The cap to check against: ``explicit`` (a library caller's value), else the env override,
    else the backend default."""
    if explicit is not None:
        if explicit < 1:
            raise ArchiveCapConfigError(f'the archive cap must be a positive whole number of bytes, got {explicit}.')
        return explicit
    raw = os.environ.get(MAX_UNCOMPRESSED_BYTES_ENV, '').strip()
    if not raw:
        return DEFAULT_MAX_UNCOMPRESSED_BYTES
    refusal = f'{MAX_UNCOMPRESSED_BYTES_ENV} must be a positive whole number of bytes, got {raw!r}.'
    try:
        value = int(raw)
    except ValueError:
        raise ArchiveCapConfigError(refusal) from None
    if value < 1:
        raise ArchiveCapConfigError(refusal)
    return value


def _human(size: int) -> str:
    return f'{size:,} bytes ({size / (1024 * 1024):.1f} MiB)'


def check_uncompressed_total(members: Sequence[tuple[str, int]], cap: int, *, remedy: str) -> int:
    """Return the total of ``members`` (``(key, size)`` pairs), or raise when it exceeds ``cap``.

    The comparison matches the server's: a total equal to the cap is accepted. The refusal names the
    total, the cap, the largest members with their sizes, the remedy, and how to change the cap.
    """
    total = sum(size for _, size in members)
    if total <= cap:
        return total
    largest = sorted(members, key=lambda member: (-member[1], member[0]))[:LARGEST_MEMBERS_SHOWN]
    listed = '\n'.join(f'  {size:>15,}  {key}' for key, size in largest)
    raise ArchiveOverCapError(
        f"the archive would expand to {_human(total)}, over the server's {_human(cap)} total-uncompressed cap. "
        'Nothing was uploaded: the server would refuse this archive, but only after receiving all of it.\n'
        f'Largest of {len(members)} file(s):\n{listed}\n'
        f'{remedy}\n'
        'If this server runs with a raised total-uncompressed cap, set '
        f'{MAX_UNCOMPRESSED_BYTES_ENV} to the same value.'
    )


__all__ = [
    'DEFAULT_MAX_UNCOMPRESSED_BYTES',
    'LARGEST_MEMBERS_SHOWN',
    'MAX_UNCOMPRESSED_BYTES_ENV',
    'ArchiveCapConfigError',
    'ArchiveOverCapError',
    'check_uncompressed_total',
    'resolve_max_uncompressed_bytes',
]
