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
import tarfile
from collections.abc import Sequence

#: The backend default total-uncompressed cap (128 MiB).
DEFAULT_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024

#: Client-side override for a server whose operator changed the cap above.
MAX_UNCOMPRESSED_BYTES_ENV = 'PLURAI_INGEST_MAX_UNCOMPRESSED_BYTES'

#: How many of the largest members the refusal names.
LARGEST_MEMBERS_SHOWN = 5

#: What each archive of a multi-part export keeps free beyond its members' weights: a global header
#: (1,024 bytes), the end-of-archive blocks and record padding (at most 10,240) and zlib's worst-case
#: growth on 128 MiB of incompressible bytes (at most 40,995 with the gzip framing).
ARCHIVE_TAIL_BYTES = 65_536

#: The most members the server reads from one archive (the scrub map counts as one).
MAX_MEMBERS_PER_ARCHIVE = 100_000

#: The most parts one grounding delivery takes; an export that needs more is refused before anything is sent.
MAX_PARTS_PER_DELIVERY = 256

_TAR_BLOCK = 512


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


def tar_member_weight(key: str, size: int) -> int:
    """The bytes a member named ``key`` of ``size`` bytes takes in the decompressed tar stream: its header
    (with the PAX record a long or non-ASCII name needs) plus its content padded to a 512-byte block.

    The header is built exactly as the archive writer builds it (PAX format, mode 0644, zero times and owner),
    so the weight is the stream's own count.
    """
    info = tarfile.TarInfo(name=key)
    info.size = size
    info.mode = 0o644
    header = len(info.tobuf(tarfile.PAX_FORMAT, tarfile.ENCODING, 'surrogateescape'))
    return header + -(-size // _TAR_BLOCK) * _TAR_BLOCK


def _member_budget(cap: int, reserve: int) -> int:
    return cap - ARCHIVE_TAIL_BYTES - reserve


def check_each_member_fits(members: Sequence[tuple[str, int]], cap: int, *, reserve: int, remedy: str) -> int:
    """Return the total size of ``members`` (``(key, size)`` pairs), or raise when one of them cannot fit an
    archive of its own.

    One archive holds at most ``cap`` bytes of tar stream, of which :data:`ARCHIVE_TAIL_BYTES` and ``reserve``
    (the weight of the member every archive also carries, the scrub map, or 0) are spoken for. A file is never
    split across archives, so a file whose weight is over what is left is refused, named with its size.
    """
    budget = _member_budget(cap, reserve)
    for key, size in sorted(members):
        if tar_member_weight(key, size) > budget:
            raise ArchiveOverCapError(
                f"{key} alone is {_human(size)}, over the server's {_human(cap)} per-archive cap once the archive's own "
                'framing is counted; a file is never split across archives. Nothing was uploaded.\n'
                f'{remedy}\n'
                'If this server runs with a raised total-uncompressed cap, set '
                f'{MAX_UNCOMPRESSED_BYTES_ENV} to the same value.'
            )
    return sum(size for _, size in members)


def pack_whole_members(
    members: Sequence[tuple[str, int]],
    cap: int,
    *,
    reserve: int,
    remedy: str,
    max_members: int = MAX_MEMBERS_PER_ARCHIVE,
    max_parts: int = MAX_PARTS_PER_DELIVERY,
) -> list[list[str]]:
    """Pack ``members`` (``(key, size)`` pairs) into archives of whole members, and return each archive's keys.

    Next-fit over the keys in code-point order (the order the archive writer stores them in): an archive closes
    when the next member's :func:`tar_member_weight` would take its sum past ``cap - ARCHIVE_TAIL_BYTES - reserve``
    (a sum equal to that budget is accepted), or when it holds ``max_members`` members, counting the reserved one
    when ``reserve`` is not 0. The result depends only on the keys, sizes and limits, so the same export packs the
    same way on every run.

    A member that cannot fit an archive of its own raises :class:`ArchiveOverCapError`
    (:func:`check_each_member_fits`); so does an export that needs more than ``max_parts`` archives, before any
    archive is built.
    """
    check_each_member_fits(members, cap, reserve=reserve, remedy=remedy)
    budget = _member_budget(cap, reserve)
    slots = max_members - (1 if reserve else 0)
    if slots < 1:
        raise ValueError(f'an archive of at most {max_members} member(s) has no room beside the reserved one')
    bins: list[list[str]] = []
    current: list[str] = []
    weight = 0
    for key, size in sorted(members):
        member = tar_member_weight(key, size)
        if current and (weight + member > budget or len(current) >= slots):
            bins.append(current)
            current, weight = [], 0
        current.append(key)
        weight += member
    if current:
        bins.append(current)
    if len(bins) > max_parts:
        total = sum(size for _, size in members)
        raise ArchiveOverCapError(
            f'the export ({len(members)} file(s), {_human(total)}) needs {len(bins)} archives of at most {_human(cap)}, '
            f'more than the {max_parts} one delivery takes. Nothing was uploaded.\n{remedy}'
        )
    return bins


__all__ = [
    'ARCHIVE_TAIL_BYTES',
    'DEFAULT_MAX_UNCOMPRESSED_BYTES',
    'LARGEST_MEMBERS_SHOWN',
    'MAX_MEMBERS_PER_ARCHIVE',
    'MAX_PARTS_PER_DELIVERY',
    'MAX_UNCOMPRESSED_BYTES_ENV',
    'ArchiveCapConfigError',
    'ArchiveOverCapError',
    'check_each_member_fits',
    'check_uncompressed_total',
    'pack_whole_members',
    'resolve_max_uncompressed_bytes',
    'tar_member_weight',
]
