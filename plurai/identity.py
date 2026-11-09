"""ONE SDK identity, computed from the LOADED module.

**The rule.** The only question about a running SDK that has a single answer is *which files is
this process importing?* That is `plurai.__file__`, and it is what this module reads. Dist
metadata (`plurai.__version__`), a pin resolved out of `direct_url.json` and a version-string
floor comparison are three other questions, and they can answer a single binary three different
ways.

**The shape.** `source` says how the identity was derived and therefore what the other fields
mean. `git` — the loaded package sits inside a checkout, so `commit` + `dirty` name it exactly and
`version` is only a label. `dist` — an installed wheel, so there is no commit and `file_hash`
(the same content digest the backend's `source_id` uses, one axis over) is what actually moves. Never
a guess: an unreadable tree — or one whose git reads did not COMPLETE, so which branch it belongs
on was never decided (`_GitRead`) — gets `source: 'unknown'` and empty fields, because a partial
identity that compares equal to another partial identity is the failure this module exists to
remove.

**The contract block** rides along so a verdict can say which CLIENT SHAPE produced the artifact
it graded — the mirror of the backend's `source_id`. `transcript_schema` is the artifact shape;
`wire_spec_pin` is the backend commit whose OpenAPI this build's wire models were generated from,
which is what decides whether a field the backend now sends survives `extra='ignore'`.

Nothing here is derived from the version string, and nothing here should be: a floor decision
belongs on a capability probe (`plurai capabilities`) or a commit date.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from plurai._platform import utf8_child_text

__all__ = ['sdk_identity', 'sdk_identity_line']

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _GitRead:
    """One git read: what it said, and WHETHER it said anything.

    ``output`` alone would mean three things at once, and the identity turns on telling them
    apart. ``ls-files --error-unmatch`` exits non-zero for a path the index does not track — a
    real ANSWER, "no" — while a call that TIMED OUT gives no answer at all. Reading the second as
    the first stamps one binary with two identities: with the 5s timeout tripped (a loaded
    machine, a slow filesystem, a huge index) the package would look untracked, the run would
    fall through to ``dist`` with an empty ``commit``, and nothing would say so.
    """

    output: str | None
    #: The call produced a VERDICT: a clean exit, a non-zero exit, or no ``git`` on this machine
    #: at all — which is itself definitive, since a package that cannot be asked about a checkout
    #: by a machine with no git is not in one this process can read. ``False`` only for a call
    #: that FAILED TO RUN (a timeout, a broken pipe), where ``output is None`` states nothing.
    answered: bool


#: A read nobody performed, because an earlier one already settled the branch. Not a failure —
#: nothing was lost — so it must not push the identity to ``unknown``.
_UNASKED = _GitRead(output=None, answered=True)


def _git(cwd: Path, *args: str) -> _GitRead:
    """One git read. Never raises, never guesses, and SAYS when it could not answer."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ['git', *args],  # noqa: S607 - git resolved via PATH by design
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            **utf8_child_text(),  # git prints UTF-8
        )
    except FileNotFoundError:
        # No git binary. Definitive, and the ``dist`` branch below is the RIGHT answer for it
        # rather than a fallback — so this is not the silent degradation the class docstring
        # describes, and it stays quiet.
        return _GitRead(output=None, answered=True)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning('sdk identity: `git %s` in %s did not complete (%s: %s).', args[0], cwd, type(exc).__name__, exc)
        return _GitRead(output=None, answered=False)
    return _GitRead(output=proc.stdout.strip() if proc.returncode == 0 else None, answered=True)


def _package_file_hash(root: Path) -> str:
    """``sha256`` over ``(relative path, bytes)`` of every ``*.py`` under ``root``, or ``''``.

    Path AND bytes, so a build that adds or renames a module moves the digest even when every
    surviving file is byte-identical; relative, so the same tree hashes the same wherever a venv
    happened to put it. An unreadable tree hashes to ``''`` — unknown, matching nothing.
    """
    digest = hashlib.sha256()
    try:
        paths = sorted(p for p in root.rglob('*.py') if p.is_file())
        if not paths:
            return ''
        for path in paths:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b'\0')
            digest.update(path.read_bytes())
            digest.update(b'\0')
    except OSError:
        return ''
    return digest.hexdigest()[:16]


def _contract() -> dict[str, Any]:
    """WHICH CLIENT SHAPE this build speaks — the artifact schema and the wire spec it was cut from."""
    from plurai.run_artifacts import TRANSCRIPT_SCHEMA  # noqa: PLC0415 - leaf read, kept lazy so importing identity costs nothing

    pin = Path(__file__).resolve().parent.parent / 'wire' / 'spec-pin.txt'
    try:
        wire_spec_pin = pin.read_text(encoding='utf-8').strip()
    except OSError:
        # An installed wheel does not ship ``wire/``; the field is EMPTY, never omitted, so a
        # reader can tell "this build did not know" from "this build was cut from spec X".
        wire_spec_pin = ''
    return {'transcript_schema': TRANSCRIPT_SCHEMA, 'wire_spec_pin': wire_spec_pin}


@cache
def _cached_identity() -> dict[str, Any]:
    """The memoised derivation — see :func:`sdk_identity`, which is what callers use. Its ``path``
    reaches the artifacts and the upload, so it is written the way everything leaving the machine
    is (:func:`plurai.host_effects.scrub_parent_record`): from ``~``, never naming the operator."""
    from plurai.host_effects import scrub_parent_record  # noqa: PLC0415 - leaf read, kept lazy like _contract

    package = Path(__file__).resolve().parent
    return scrub_parent_record(_identity_for(package), repo=package)


def sdk_identity() -> dict[str, Any]:
    """WHICH SDK is running, derived from :data:`plurai.__file__`. Derived once per process.

    Memoised because it shells out to git and can walk the package tree: this is stamped into
    ``run_summary.json``, every ``transcript.json.metadata`` and the ingest provenance, and
    several subprocess round-trips per scenario would be paid for an answer that cannot change
    inside one process. Returns a plain dict — it is JSON on the way to an artifact, and a model
    here would make every reader import the SDK to read a run it already has on disk.

    A FRESH copy each call, ``contract`` included. The memoised value is stamped into every
    artifact of the run, so handing every caller the same mutable nested dict makes one caller's
    edit retroactively true of every artifact written after it.
    """
    return deepcopy(_cached_identity())


def _identity_for(package: Path) -> dict[str, Any]:
    """:func:`sdk_identity` for an arbitrary package directory — the testable half.

    Split out so the git branch can be exercised against a real repository: reading the LOADED
    module is the whole point of this module, and a test that can only see whichever branch the
    developer's own checkout happens to take proves nothing about the other one.
    """
    from plurai import __version__  # noqa: PLC0415 - the package imports this module's callers; lazy keeps the cycle closed

    identity: dict[str, Any] = {
        'source': 'unknown',
        'version': __version__,
        'path': str(package),
        'commit': '',
        'dirty': False,
        'file_hash': '',
        'contract': _contract(),
    }
    if not package.is_dir():  # pragma: no cover - a zipimported package has no directory
        return identity

    # A CHECKOUT the loaded package lives inside AND IS TRACKED BY. `rev-parse` alone answers for
    # any directory that has a repo somewhere above it and never consults `.gitignore`, so the
    # standard uv/venv layout — the package at `<customer-repo>/.venv/lib/pythonX/site-packages/
    # plurai` — would hand back the CUSTOMER repo's HEAD as the SDK's commit, with
    # `dirty: False` (that path is gitignored, so nothing is ever reported dirty) and an empty
    # `file_hash`, because the correct `dist` branch below would be unreachable. `ls-files
    # --error-unmatch` is the question that separates "a repo is above me" from "this repo is
    # what I am": it exits non-zero when the pathspec matches nothing the index tracks.
    head = _git(package, 'rev-parse', 'HEAD')
    tracked = _git(package, 'ls-files', '--error-unmatch', '--', str(package)) if head.output is not None else _UNASKED
    # `--porcelain` over the PACKAGE directory, not the repo root: an SDK checkout that is
    # dirty only in its own tests or docs is not a different runner, and flagging it as one
    # is how a dirty marker stops meaning anything.
    status = _git(package, 'status', '--porcelain', '--', str(package)) if tracked.output is not None else _UNASKED
    if head.output is not None and tracked.output is not None and status.output is not None:
        return {**identity, 'source': 'git', 'commit': head.output, 'dirty': bool(status.output)}
    if not (head.answered and tracked.answered and status.answered):
        # A git read that never completed. Falling through to `dist` here is a GUESS about the
        # question that was not answered — and a guess that flips with machine load, so two runs
        # of one binary stamp two identities: `source: git` with a commit when the reads are
        # fast, `source: dist` with an empty commit when one times out. This module's own rule is
        # that an identity it cannot derive is `unknown`, precisely so a partial answer never
        # compares equal to a real one.
        logger.warning(
            'sdk identity: a git read for %s did not complete, so whether this package sits in a checkout that TRACKS it '
            'was never decided. The identity is reported as UNKNOWN rather than derived from the reads that did return — '
            'two runs of one binary must not stamp two identities.',
            package,
        )
        return identity
    if (file_hash := _package_file_hash(package)) != '':
        return {**identity, 'source': 'dist', 'file_hash': file_hash}
    return identity


def sdk_identity_line() -> str:
    """The one-line human rendering, shared by every caller that prints it.

    One spelling, because two renderings of one identity is how a run log and a pin come to
    disagree about the binary they both resolved.
    """
    ident = sdk_identity()
    if ident['source'] == 'git':
        return f'plurai {ident["version"]} @ {ident["commit"][:12]}{"-dirty" if ident["dirty"] else ""} ({ident["path"]})'
    if ident['source'] == 'dist':
        return f'plurai {ident["version"]} dist:{ident["file_hash"]} ({ident["path"]})'
    return f'plurai {ident["version"]} (identity UNKNOWN — {ident["path"]} could not be resolved to a checkout or a distribution)'
