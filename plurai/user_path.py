"""Expansion of a leading ``~`` in a path given on a command line."""

from __future__ import annotations

import os


def expand_user_path(raw: str) -> str:
    """``raw`` with a leading ``~`` / ``~user`` expanded to a home directory.

    A shell expands ``~`` only when it starts an unquoted word, so a quoted value,
    ``--flag=~/x`` or a value read from a config or env var reaches the process with the
    literal ``~``. Expanding at the argparse boundary means no later code sees one. Never
    raises: a ``~user`` that does not exist, or a ``~`` with no home to resolve, comes back
    unchanged. The result is not normalized. Returns a ``str`` so the parsed namespace keeps
    the type it always had.
    """
    return os.path.expanduser(raw)  # noqa: PTH111 - Path.expanduser() raises where this returns the input
