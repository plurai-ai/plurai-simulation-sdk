"""Files only their owner may read, written the Windows way.

POSIX gets this from file modes and directory descriptors (``plurai.credentials`` keeps that path as it is). Windows has
neither ``dir_fd`` nor modes, so the folder's access list is cut to the current user first; the file is then written to a
random temporary name inside it, its access list is cut to the user *before* the secret is written to it, and
``os.replace`` puts it in place. The secret is never in a file other accounts can open, and a reader never sees half a file.
A symlink or any other reparse point at the file is refused rather than followed.

An access list is set and read back on a handle opened on the object itself without following links
(:func:`plurai._winsec.restrict`), so a path swapped for a junction after the caller looked at it changes nothing. When the
list cannot be set, or reads back as anything but the user alone, the write fails: a secret is not left under the folder's
inherited access. No command is run.

Stdlib only. The platform is the caller's decision; the Windows calls are in :mod:`plurai._winsec`.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import time
from pathlib import Path

from plurai import _winsec
from plurai._platform import is_reparse_point

#: Waits (seconds) between tries to move the file into place: a hook, an antivirus scan or an indexer holds the old one open.
_REPLACE_WAITS = (0.1, 0.2, 0.4, 0.8, 1.6)


class PrivateFileError(OSError):
    """The file cannot be written, or cannot be limited to the user. ``str()`` says why and never holds file content."""


def current_sid() -> str:
    """The SID of the user running this process, from the process token; ``PrivateFileError`` when it cannot be read."""
    try:
        return _winsec.user_sid()
    except OSError as exc:
        raise PrivateFileError(f'the current user cannot be identified ({exc})') from exc


def restrict_to_user(path: Path, *, directory: bool = False, require_owner: bool = False) -> None:
    """Cut the access list of ``path`` to the current user (inheritance removed), then check it reads back as the user alone.

    With ``require_owner`` an object owned by an account other than the user or the Administrators group is refused before its
    list is changed. Raises ``PrivateFileError`` for a link, a foreign owner, or a list that cannot be set.
    """
    user = current_sid()
    try:
        security = _winsec.restrict(path, user, directory=directory, owners=(user, _winsec.ADMINISTRATORS_SID) if require_owner else None)
    except _winsec.ReparsePointError as exc:
        raise PrivateFileError(f'{path} is a symlink or junction, which is not followed') from exc
    except _winsec.OwnerError as exc:
        raise PrivateFileError(f'{_winsec.message("owner", path)}; fix its owner, or delete {path} and try again') from exc
    except OSError as exc:
        raise PrivateFileError(f'{path} cannot be limited to your user ({exc})') from exc
    if _winsec.restricted_problem(security.allowed, security.dacl_present, user):
        raise PrivateFileError(_winsec.message('not_restricted', path))


def restrict_directory(path: Path) -> None:
    """Make ``path`` (a folder) the user's alone, as a folder of mode 0700 is on POSIX: a link or a folder another account owns is refused."""
    if is_reparse_point(os.lstat(path)):
        raise PrivateFileError(f'{path} is a symlink or junction, which is not followed')
    restrict_to_user(path, directory=True, require_owner=True)


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` as ``path``: a random temporary name beside it, limited to the user before ``data`` goes in, then ``os.replace``.

    Raises ``PrivateFileError`` for a reparse point at ``path`` or an access list that cannot be set, and ``OSError`` for
    the filesystem's own failures. The temporary file is removed on every failure.
    """
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing is not None and is_reparse_point(existing):
        raise PrivateFileError(f'{path.name} is a symlink or junction, which is not followed; remove it first')
    temporary = path.with_name(f'.{path.name}.{secrets.token_hex(8)}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    try:
        try:
            restrict_to_user(temporary)
            with os.fdopen(descriptor, 'wb') as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor != -1:
                os.close(descriptor)
        for wait in _REPLACE_WAITS:
            try:
                temporary.replace(path)
                break
            except PermissionError:
                time.sleep(wait)
        else:
            temporary.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
