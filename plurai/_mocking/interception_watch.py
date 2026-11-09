"""Whether the files that route the agent's vendor calls to the mock stayed as written.

On the ``code`` action channel the interception lives in files the agent's own shell can reach: the
``sitecustomize.py`` that installs the import shadow, the surface it reads, the Node preload. An agent
that rewrites one switches the interception off, and its later vendor calls go to the real service
while the run still scores the world the mock kept. Nothing here prevents that: the harness records
each file's content when it writes it (:func:`watch_interception`), and the child compares after every
turn and at finalize (:func:`interception_changed`), so the run reports the change instead of scoring
over it in silence.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: Per watched root, ``{name: sha256}`` of every file under it as recorded. A file root names itself.
_WATCHED: dict[Path, dict[str, str]] = {}
#: Every name seen changed so far in this process. A change is never forgotten: an agent that
#: restores the file afterwards still ran turns without the interception.
_CHANGED: set[str] = set()


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _snapshot(root: Path) -> dict[str, str]:
    """``{name: sha256}`` for ``root``: the file itself, or every regular file under the directory by its
    path relative to it. A file that cannot be read is absent, which reads as removed. ``__pycache__``
    is not read: the interpreter writes it beside a module it imports, and no agent wrote it."""
    if not root.is_dir():
        digest = _digest(root)
        return {root.name: digest} if digest is not None else {}
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_file() and (digest := _digest(path)) is not None:
                found[path.relative_to(root).as_posix()] = digest
    return found


def watch_interception(*paths: str | os.PathLike[str]) -> None:
    """Record the content of each file or directory in ``paths`` as it stands now.

    Called by the harness right after it writes the interception (:func:`~plurai.write_sitecustomize`
    records its own file). Recording a path again replaces its record, for a harness that rewrites it.
    """
    for path in paths:
        root = Path(path).resolve()
        _WATCHED[root] = _snapshot(root)


def interception_changed() -> list[str]:
    """The watched names whose content differs from the record — changed, removed, or a file added
    under a watched directory — accumulated over every call in this process, sorted. ``[]`` when
    nothing was watched or nothing changed."""
    for root, recorded in _WATCHED.items():
        current = _snapshot(root)
        _CHANGED.update(name for name in recorded.keys() | current.keys() if recorded.get(name) != current.get(name))
    return sorted(_CHANGED)


def interception_watched() -> bool:
    """Whether this process recorded any interception file: a process that recorded none claims nothing."""
    return bool(_WATCHED)
