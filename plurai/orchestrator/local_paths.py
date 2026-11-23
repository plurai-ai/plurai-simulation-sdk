"""Which whole strings name a place on the operator's machine, and what stands in for one on the wire.

Detection uses ``pathlib``'s pure classes only: no filesystem read, so the answer is the same on every
machine and for every run. It fails closed for absolute paths: ``/api/orders/42`` and ``/Users/x/proj`` have the same
shape, so a caller that applies this to a bookkeeping value accepts that a route is replaced too.

It imports only the standard library, so any record writer can use the marker.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

#: What a local path is replaced by. It holds no digit, so a replaced path never reads as a status code or a
#: size, and it does not start with the size descriptor prefix, which marks a value the agent needs.
LOCAL_PATH_MARKER: Final = '[plurai: local path not sent]'

#: A ``file:`` URL with an absolute path: ``file:///x``, ``file://localhost/x`` and the one-slash ``file:/x`` (RFC 8089).
_FILE_URL_PREFIX: Final = 'file:/'

#: Quotes a harness may leave around a value it formatted (``repr`` of a path, a shell-quoted argument).
_QUOTES: Final = ('"', "'")


def is_local_path(value: str) -> bool:
    """Whether the whole of ``value`` is a POSIX absolute, home-relative, Windows drive, UNC or ``file:`` path.

    A single line only. Whitespace around the value (the newline a command prints after a folder name included) and
    one pair of matching quotes around it are not part of it, so ``' /Users/x'``, ``'/Users/x\\n'`` and
    ``'"/Users/x"'`` are paths. A newline inside the value makes it text. Full URLs (``https://…``, ``s3://…``),
    relative paths (``.``, ``./out``), ``C:relative`` and a sentence that contains a path are not local paths.
    """
    value = value.strip()
    if '\n' in value or '\r' in value:
        return False
    if len(value) > 1 and value[0] in _QUOTES and value[-1] == value[0]:
        value = value[1:-1].strip()
    if not value:
        return False
    if value.lower().startswith(_FILE_URL_PREFIX):
        return True
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute():
        return True
    # ``~``, ``~/x``, ``~name/x``, ``~\\x``: both pure classes keep a leading ``~`` in the first part, and ``.`` has no parts.
    return value.startswith('~')
