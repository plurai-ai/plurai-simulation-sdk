"""Whole-file writes: a reader of the destination sees the old file or the new one, never half of either."""

from __future__ import annotations

import contextlib
import os
import secrets
from pathlib import Path


def write_atomic(dest: Path, content: bytes | str) -> None:
    """Write ``content`` to ``dest`` through a temporary file beside it and a rename. ``str`` content is written as UTF-8.

    The temporary name is unique per writer, so two writers of one destination never rename each other's half-written file.
    The file gets the default mode, because a later reader (another user, a CI step) needs to read it; a file that must stay
    private is written by :func:`plurai._private_files.write_atomic`. The parent directory must exist. On any failure the
    temporary file is removed and ``dest`` is untouched.

    Raises:
        OSError: from the write or the rename.
    """
    data = content.encode('utf-8') if isinstance(content, str) else content
    tmp = dest.with_name(f'.{dest.name}.{os.getpid()}-{secrets.token_hex(4)}.tmp')
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


__all__ = ['write_atomic']
