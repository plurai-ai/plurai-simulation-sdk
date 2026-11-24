"""The few read-only git questions the CI commands ask about the checkout they run in."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import NamedTuple

_TIMEOUT_SECONDS = 30


class Answer(NamedTuple):
    ok: bool
    stdout: str


def git(workdir: Path, *args: str) -> Answer:
    """``git -C workdir args``. Not ok when git is missing, times out or exits non-zero."""
    try:
        done = subprocess.run(['git', '-C', str(workdir), *args], capture_output=True, text=True, check=False, timeout=_TIMEOUT_SECONDS)  # noqa: S603, S607
    except (OSError, subprocess.SubprocessError):
        return Answer(False, '')
    return Answer(done.returncode == 0, done.stdout)


def head_commit(workdir: Path, *, short: bool = False) -> str | None:
    """The checkout's HEAD commit, or ``None`` outside a repository."""
    answer = git(workdir, 'rev-parse', *(['--short=12'] if short else []), 'HEAD')
    return answer.stdout.strip() or None if answer.ok else None


def merge_base_with_default_branch(workdir: Path) -> str | None:
    """The merge-base of HEAD with the default branch, for a developer who names no ref."""
    for candidate in ('origin/HEAD', 'origin/main', 'main', 'origin/master', 'master'):
        answer = git(workdir, 'merge-base', 'HEAD', candidate)
        if answer.ok and answer.stdout.strip():
            return answer.stdout.strip()
    return None
