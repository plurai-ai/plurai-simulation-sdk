"""``plurai diagnose``: the bundle a user pastes into a support issue.

    plurai diagnose [--workdir DIR] [--limit N]

Plurai clients send no telemetry, so support learns what happened only from what the user pastes. This command prints
that in one block: the versions in play, whether a credential and a backend url are present, and the identifiers a
support engineer can look up on the backend (command run id, error ids, run names).

What the bundle holds is decided by construction, not by filtering a dump. Every line is built from a value this module
names: a version string, a presence flag, a url origin, an identifier that matches a fixed shape, a status word. A file
of the user's project is never read for its content; the run folder is read for the few fields listed in
:data:`_ERROR_FIELDS` and :data:`_SUMMARY_FIELDS`, each of which is an id, a class name or a number. The finished text
then passes through :func:`plurai.host_effects.redact_cli_credentials`, with the saved token registered first, so a value
that reached the text by a path nobody foresaw is still replaced.

Local references. A client that cannot reach the backend (a failed install, a hook decision that refused a call) has no
backend id to quote, so it appends one line to ``~/.plurai/support_refs.jsonl`` instead::

    {"at": "2026-01-01T12:00:00Z", "kind": "install_failure", "ref": "inst-1a2b3c", "summary": "one short line"}

``kind`` and ``ref`` are short tokens; ``summary`` is customer-safe text the writer composed. The bundle prints the newest
lines, each field validated and scrubbed with :func:`plurai.customer_safe.safe_text`.

An SDK that did not install cannot run this command. The help text lists what to paste in its place.

Exit 0 when a bundle was printed; 64 for a usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import stat
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

from plurai import __version__, client_handshake, command_run, credentials, exit_codes
from plurai._endpoints import TOKEN_ENV, URL_ENV
from plurai._platform import is_reparse_point, is_windows

DISTRIBUTION = client_handshake.CLIENT_NAME

#: Beside the credentials file, under the same private directory.
REFERENCES_FILENAME = 'support_refs.jsonl'

#: The environment variable the command run id reaches a child process as.
COMMAND_RUN_ID_ENV = command_run.ENV_VAR

#: The pointer file the runner writes beside ``simulate.py``: the absolute ``--out`` of the last run.
_LAST_RUN = '.last_run'
#: The folder the generated harness lives in, under the workspace root.
_HARNESS_DIRNAME = 'plurai_sim'
#: How many folders below the workspace root the ``plurai_sim`` folder itself may sit, counting it: ``a/b/c/plurai_sim``
#: is found, ``a/b/c/d/plurai_sim`` is not. The search lists at most this many folders in total, shallowest first.
_HARNESS_DEPTH = 4
_VISIT_LIMIT = 5000
#: Folders the search does not enter: version control, dependency, tool-cache and Plurai's own state folders. The
#: plugin's harness tooling skips the same names.
_NOT_SEARCHED = frozenset({'.git', '.hg', '.venv', 'venv', 'node_modules', '__pycache__', '.tox', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.plurai'})

#: Fields read from a scenario's ``error.json``: a class name, a typed kind, the backend's error id and an HTTP status.
#: Never the message or the request url, which can quote the user's data.
_ERROR_FIELDS = ('error_id', 'kind', 'exception', 'status')
#: Fields read from ``run_summary.json``: the run's name and when it was written, and its counts.
_SUMMARY_FIELDS = ('run_name', 'written_at', 'total', 'succeeded', 'failed', 'incomplete', 'exit_code')

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}')
_WORD = re.compile(r'[a-z][a-z0-9_-]{0,39}')
_STAMP = re.compile(r'[0-9T:+\-.Z]{10,40}')
_NUMBER_FIELDS = frozenset({'total', 'succeeded', 'failed', 'incomplete', 'exit_code', 'status'})

_READ_LIMIT = 256 * 1024
_REFERENCE_SUMMARY_CHARS = 200
_MAX_ERROR_FILES = 200
DEFAULT_LIMIT = 10

INSTALL_FAILURE_HELP = """\
If `pip install plurai-simulation-sdk` fails, this command cannot run. Paste these instead:
  - the output of `pip install -v plurai-simulation-sdk` (look it over first: remove any index url that carries a password);
  - `python --version` and `pip --version`;
  - your operating system and CPU (`uname -a` on macOS or Linux, `ver` on Windows);
  - the harness you use the plugin in, and the plugin version.
Never paste the contents of ~/.plurai/credentials.json or a PLURAI_TOKEN."""

EPILOG = f"""\
What the bundle holds: versions, presence of a credential and a backend url (never their values), the url's origin,
the command run id, recent error ids and run names, and the references other clients recorded locally. It holds no
token and no file from your project.

{INSTALL_FAILURE_HELP}
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--workdir',
        type=Path,
        default=None,
        help=(
            'the workspace to look for the latest run in (default: the current directory). It reads plurai_sim/ at the workspace root when it holds a run, '
            'else the plurai_sim/ folder up to four folders below it (a/b/c/plurai_sim), skipping symlinks and dependency or cache folders; '
            'with several, the one whose .last_run was written last. With several agents, pass the agent folder. '
            'Only ids and counts are read from it'
        ),
    )
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT, help=f'how many recent error ids and references to list (default {DEFAULT_LIMIT})')


def _origin(url: str) -> str:
    """``scheme://host[:port]`` of ``url``, which drops userinfo, path and query; a value that is not a url reads ``unreadable``."""
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return 'unreadable'
    if parts.scheme not in ('http', 'https') or not host:
        return 'unreadable'
    host = f'[{host}]' if ':' in host else host
    return f'{parts.scheme}://{host}' + (f':{port}' if port else '')


def installed_extras(distribution: str = DISTRIBUTION) -> list[str]:
    """The extras of ``distribution`` whose own requirements are all installed, sorted."""
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return []
    declared = dist.metadata.get_all('Provides-Extra') or []
    required: dict[str, list[str]] = {name: [] for name in declared}
    for line in dist.requires or []:
        marker = re.search(r"""extra\s*==\s*["']([^"']+)["']""", line)
        name = re.match(r'[A-Za-z0-9][A-Za-z0-9._-]*', line)
        if marker and name and marker.group(1) in required and name.group(0).lower().replace('_', '-') != distribution:
            required[marker.group(1)].append(name.group(0))
    present = []
    for extra, names in required.items():
        if names and all(_is_installed(name) for name in names):
            present.append(extra)
    return sorted(present)


def _is_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _credential_lines(env: Mapping[str, str]) -> list[str]:
    lines: list[str] = []
    moved = credentials.home_problem(env)
    saved = credentials.Credential() if moved else credentials.read()
    if saved.token:
        from plurai.host_effects import register_backend_token

        register_backend_token(saved.token)
    if moved:
        lines.append(
            'Credentials file: not looked for (the user profile cannot hold a saved credential; see `plurai login --status`)'
            if is_windows()
            else 'Credentials file: not looked for (HOME is not the account home)'
        )
    elif saved.token:
        lines.append(f'Credentials file: present (backend {_origin(saved.url)})')
    elif saved.problem:
        lines.append('Credentials file: present but unusable (run `plurai login --status` for the reason)')
    else:
        lines.append('Credentials file: absent')
    url = (env.get(URL_ENV) or '').strip()
    lines.append(f'PLURAI_URL: present (origin {_origin(url)})' if url else 'PLURAI_URL: absent')
    lines.append('PLURAI_TOKEN: present' if (env.get(TOKEN_ENV) or '').strip() else 'PLURAI_TOKEN: absent')
    return lines


def _command_run_id_line(env: Mapping[str, str]) -> str:
    value = (env.get(COMMAND_RUN_ID_ENV) or '').strip()
    if not value:
        return f'Command run id ({COMMAND_RUN_ID_ENV}): not set in this shell'
    if not _ID.fullmatch(value):
        return f'Command run id ({COMMAND_RUN_ID_ENV}): set, but not an id'
    return f'Command run id ({COMMAND_RUN_ID_ENV}): {value}'


def _open_no_follow(path: Path) -> int:
    """A read descriptor for ``path``, never through a symlink. Windows has no ``O_NOFOLLOW``: a reparse point is refused first."""
    judged = os.lstat(path) if is_windows() else None
    if judged is not None and is_reparse_point(judged):
        raise OSError(f'{path} is a symlink or junction')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_BINARY', 0))
    if judged is not None:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (judged.st_dev, judged.st_ino):  # swapped for a link between the check and the open
            os.close(descriptor)
            raise OSError(f'{path} changed while it was being opened')
    return descriptor


def _read_json_object(path: Path) -> dict[str, object] | None:
    """The JSON object in ``path`` (a regular file, read up to a bound, never through a symlink), or ``None``."""
    try:
        descriptor = _open_no_follow(path)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        raw = os.read(descriptor, _READ_LIMIT)
    finally:
        os.close(descriptor)
    try:
        document = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _pointed_run(base: Path) -> Path | None:
    """The run folder ``base/.last_run`` points at, or ``None`` when it has no usable pointer."""
    try:
        pointer = base / _LAST_RUN
        if pointer.is_symlink() or not pointer.is_file():
            return None
        with pointer.open(encoding='utf-8') as handle:
            text = handle.read(4096).strip()
        run = Path(text) if text else None
        return run if run is not None and run.is_dir() else None
    except (OSError, ValueError):
        return None


def _is_real_dir(path: Path) -> bool:
    """A directory that is not a symlink; an unreadable path is not one."""
    try:
        return not path.is_symlink() and path.is_dir()
    except OSError:
        return False


def _pointer_stamp(folder: Path) -> int | None:
    """The modification time of ``folder/.last_run`` in nanoseconds, or ``None`` when it holds no usable run pointer."""
    try:
        return (folder / _LAST_RUN).stat().st_mtime_ns if _pointed_run(folder) is not None else None
    except OSError:
        return None


def _harness_folders(workdir: Path) -> list[Path]:
    """The ``plurai_sim/`` folders of the workspace ``workdir``, best first.

    The scan puts the harness beside the agent module (``src/app/plurai_sim``), and in the workspace root when the agent
    sits there. ``workdir/plurai_sim`` wins when its ``.last_run`` points at a run folder that exists; an empty one does
    not end the search. Otherwise the search goes below ``workdir`` one level at a time, ``_HARNESS_DEPTH`` levels down:
    a ``plurai_sim`` that is ``_HARNESS_DEPTH`` folders below ``workdir``, counting itself, is found, one further down
    is not. Within a level the folders take turns by parent, and at most ``_VISIT_LIMIT`` folders are listed, so a large
    early folder cannot hide a shallow harness. No symlink is followed (a symlinked ``plurai_sim`` is not a harness),
    ``_NOT_SEARCHED`` names are not entered, ``plurai_sim/`` itself is not entered and an unreadable folder is skipped.

    The folders whose ``.last_run`` points at a run folder that exists rank first, the one written last first; the others
    follow. A tie goes to the shallowest, then to the first by path.
    """
    root = workdir / _HARNESS_DIRNAME
    found: list[tuple[int, tuple[str, ...], Path]] = []
    if _is_real_dir(root):
        if _pointed_run(root) is not None:
            return [root]
        found.append((0, (), root))
    groups: list[list[Path]] = [[workdir]]
    visited = 0
    for depth in range(1, _HARNESS_DEPTH + 1):
        below: list[list[Path]] = []
        for folder in (child for row in itertools.zip_longest(*groups) for child in row if child is not None):
            if visited >= _VISIT_LIMIT:
                break
            visited += 1
            try:
                names = sorted(entry.name for entry in folder.iterdir())
            except OSError:
                continue
            harness = folder / _HARNESS_DIRNAME
            if _HARNESS_DIRNAME in names and _is_real_dir(harness):
                found.append((depth, harness.relative_to(workdir).parts, harness))
            below.append([folder / name for name in names if name != _HARNESS_DIRNAME and name not in _NOT_SEARCHED and _is_real_dir(folder / name)])
        groups = below
    return sorted((path for _, _, path in sorted(found)), key=lambda folder: ((stamp := _pointer_stamp(folder)) is None, -(stamp or 0)))


def _run_folder(workdir: Path) -> Path | None:
    """The run folder ``.last_run`` points at: in the best-ranked harness folder, else in ``workdir`` itself."""
    for base in (*_harness_folders(workdir), workdir):
        if (run := _pointed_run(base)) is not None:
            return run
    return None


def _clean_field(name: str, value: object) -> str | None:
    """``value`` as the bundle prints it, or ``None`` when it has not the shape its field allows."""
    if name in _NUMBER_FIELDS:
        return str(value) if isinstance(value, int) and not isinstance(value, bool) else None
    if not isinstance(value, str):
        return None
    if name == 'written_at':
        return value if _STAMP.fullmatch(value) else None
    if name in ('kind', 'exception'):
        return value if _WORD.fullmatch(value.lower()) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}', value) else None
    return value if _ID.fullmatch(value) else None


def _pick(document: Mapping[str, object], fields: Iterable[str]) -> dict[str, str]:
    picked = {}
    for name in fields:
        if name in document and (shown := _clean_field(name, document[name])) is not None:
            picked[name] = shown
    return picked


def _latest_run(workdir: Path, limit: int) -> tuple[dict[str, str], list[dict[str, str]]]:
    folder = _run_folder(workdir)
    if folder is None:
        return {}, []
    summary = _pick(_read_json_object(folder / 'run_summary.json') or {}, _SUMMARY_FIELDS)
    errors: list[dict[str, str]] = []
    try:
        scenario_dirs = sorted(entry for entry in folder.iterdir() if entry.is_dir())[:_MAX_ERROR_FILES]
    except OSError:
        scenario_dirs = []
    for scenario in scenario_dirs:
        picked = _pick(_read_json_object(scenario / 'error.json') or {}, _ERROR_FIELDS)
        if picked:
            errors.append(picked)
    return summary, errors[:limit]


def _run_lines(workdir: Path, limit: int) -> list[str]:
    summary, errors = _latest_run(workdir, limit)
    if not summary and not errors:
        return ['Latest run: none found (looked for .last_run in plurai_sim/ at or below the workdir)']
    lines = ['Latest run: ' + ', '.join(f'{key}={value}' for key, value in summary.items())] if summary else ['Latest run: no run_summary.json']
    if errors:
        lines.append('Recent errors:')
        lines.extend('  - ' + ', '.join(f'{key}={value}' for key, value in error.items()) for error in errors)
    else:
        lines.append('Recent errors: none recorded in the latest run')
    return lines


def references_path() -> Path | None:
    """The local references file, or ``None`` when the home may not be read: on Windows a profile that is a network path, on a
    drive that is not fixed, or unknown holds no file this reads."""
    if is_windows() and credentials.home_problem():
        return None
    return credentials.account_home() / '.plurai' / REFERENCES_FILENAME


def _reference_lines(limit: int) -> list[str]:
    from plurai.customer_safe import safe_text

    path = references_path()
    if path is None:
        return ['Local references: none']
    try:
        descriptor = _open_no_follow(path)
    except OSError:
        return ['Local references: none']
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            return ['Local references: none']
        size = info.st_size
        os.lseek(descriptor, max(0, size - _READ_LIMIT), os.SEEK_SET)
        raw = os.read(descriptor, _READ_LIMIT)
    finally:
        os.close(descriptor)
    rows: list[str] = []
    for line in reversed(raw.decode('utf-8', errors='replace').splitlines()):
        with contextlib.suppress(ValueError):
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            kind, ref, at = row.get('kind'), row.get('ref'), row.get('at')
            if not (isinstance(kind, str) and _WORD.fullmatch(kind) and isinstance(ref, str) and _ID.fullmatch(ref)):
                continue
            stamp = at if isinstance(at, str) and _STAMP.fullmatch(at) else 'unknown time'
            summary = row.get('summary')
            text = ' '.join(safe_text(summary).split())[:_REFERENCE_SUMMARY_CHARS] if isinstance(summary, str) else ''
            rows.append(f'  - {stamp} {kind} {ref}' + (f': {text}' if text else ''))
            if len(rows) >= limit:
                break
    return ['Local references (newest first):', *rows] if rows else ['Local references: none']


def render(workdir: Path, *, env: Mapping[str, str] | None = None, limit: int = DEFAULT_LIMIT) -> str:
    """The bundle for ``workdir`` as the text a user pastes."""
    environment = os.environ if env is None else env
    extras = installed_extras()
    lines = [
        '--- plurai diagnostic bundle (no token, no credential value, no project file content) ---',
        f'{DISTRIBUTION}: {__version__}',
        f'Extras installed: {", ".join(extras) if extras else "none"}',
        f'Python: {platform.python_version()} ({platform.python_implementation()})',
        f'OS: {platform.system()} {platform.release()} {platform.machine()}',
        *_credential_lines(environment),
        _command_run_id_line(environment),
        *_run_lines(workdir, limit),
        *_reference_lines(limit),
        '--- end of bundle ---',
    ]
    from plurai.host_effects import redact_cli_credentials

    return redact_cli_credentials('\n'.join(lines))


def run(args: argparse.Namespace) -> int:
    if args.limit < 1:
        print('plurai diagnose: --limit must be at least 1', file=sys.stderr)
        return exit_codes.USAGE
    workdir = (args.workdir or Path.cwd()).expanduser()
    print(render(workdir, limit=args.limit))
    return exit_codes.OK
