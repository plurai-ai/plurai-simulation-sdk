"""The few places the SDK behaves differently on Windows, behind one predicate.

Stdlib only. Every helper takes the platform as an optional parameter (``is_win``) so a test runs
the Windows branch on Linux without touching ``os.name``; left out, it is read from ``os.name``.
On macOS and Linux each helper is a no-op.
"""

from __future__ import annotations

import codecs
import os
import sys
from typing import TypedDict


class TextArgs(TypedDict, total=False):
    """The text-mode ``subprocess`` arguments :func:`utf8_child_text` may set."""

    encoding: str
    errors: str


def is_windows(name: str | None = None) -> bool:
    """Whether ``name`` (default ``os.name``) is Windows."""
    return (os.name if name is None else name) == 'nt'


def _resolve(is_win: bool | None) -> bool:
    return is_windows() if is_win is None else is_win


def _is_utf8(encoding: str | None) -> bool:
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == 'utf-8'
    except LookupError:
        return False


def utf8_stdio(*, is_win: bool | None = None) -> None:
    """On Windows, make stdout and stderr write UTF-8 (``errors='replace'``) when they do not already.

    Python writes a pipe or a file in the ANSI code page there (cp1252), and the SDK prints characters
    outside it (``→``, ``⇒``, ``−``), so an entry point would die on ``UnicodeEncodeError``; the
    processes reading SDK output decode it as UTF-8. A stream that is already UTF-8, or that cannot be
    reconfigured (a test's capture), is left alone.
    """
    if not _resolve(is_win):
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None or _is_utf8(getattr(stream, 'encoding', None)):
            continue
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            continue  # a closed or detached stream keeps what it had


def utf8_child_text(*, is_win: bool | None = None) -> TextArgs:
    """``subprocess`` keyword arguments for reading a child whose output is UTF-8 (the SDK's own
    modules, which call :func:`utf8_stdio`; git; node): ``encoding='utf-8', errors='replace'`` on
    Windows, where the default would decode it as the ANSI code page, and nothing elsewhere."""
    return TextArgs(encoding='utf-8', errors='replace') if _resolve(is_win) else TextArgs()


def is_reparse_point(info: os.stat_result) -> bool:
    """Whether an ``lstat`` result is a symlink or, on Windows, any reparse point (a junction included)."""
    import stat  # noqa: PLC0415 - only this predicate needs it

    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
