"""Checks of a Python package for the package workflow, using only the standard library.

``content DIST_DIR --package NAME``  the built wheel and sdist carry the import package and nothing internal.
``smoke --distribution D --module M``  run inside the environment that holds the installed package: the installed
version is the expected one, the module imports, and every console script and module entrypoint answers ``--help``.

Problems are printed to stderr, one per line, and the exit status is 1.
"""

import argparse
import importlib
import importlib.metadata
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# A path segment that never belongs in a distribution: tests, repository tooling and local state.
INTERNAL_SEGMENTS = {'tests', 'test', 'docs', 'release', 'scripts', 'evals', '.github', '__pycache__'}
INTERNAL_NAMES = {'CLAUDE.md', 'CONTEXT.md', 'AGENTS.md', 'uv.lock', '.leak-allow', 'conftest.py', 'justfile', '.gitignore'}
INTERNAL_SUFFIXES = ('.pyc', '.pem', '.key')


def _internal(member: str) -> bool:
    parts = member.split('/')
    return any(part in INTERNAL_SEGMENTS or part in INTERNAL_NAMES or part.startswith('.env') for part in parts) or member.endswith(INTERNAL_SUFFIXES)


def _one(dist: Path, pattern: str, kind: str, problems: list[str]) -> Path | None:
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        problems.append(f'expected exactly one {kind} in {dist}, found {len(found)}')
        return None
    return found[0]


def check_content(dist: Path, package: str) -> list[str]:
    problems: list[str] = []
    wheel = _one(dist, '*.whl', 'wheel', problems)
    sdist = _one(dist, '*.tar.gz', 'sdist', problems)
    if wheel:
        for name in zipfile.ZipFile(wheel).namelist():
            if not name.startswith(f'{package}/') and '.dist-info/' not in name:
                problems.append(f'{wheel.name}: {name} is outside the {package} package')
            elif _internal(name):
                problems.append(f'{wheel.name}: {name} is an internal file')
    if sdist:
        with tarfile.open(sdist) as tar:
            names = tar.getnames()
        tops = {name.split('/')[0] for name in names}
        if len(tops) != 1:
            problems.append(f'{sdist.name}: expected one top-level directory, found {sorted(tops)}')
        for name in names:
            relative = name.partition('/')[2]
            if relative and _internal(relative):
                problems.append(f'{sdist.name}: {relative} is an internal file')
    return problems


def check_smoke(args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    installed = importlib.metadata.version(args.distribution)
    if args.version and installed != args.version:
        problems.append(f'{args.distribution} is {installed}, expected {args.version}')
    try:
        importlib.import_module(args.module)
    except Exception as exc:  # any import failure is the finding
        problems.append(f'import {args.module} failed: {exc!r}')
    bin_dir = Path(args.bin_dir) if args.bin_dir else Path(sys.executable).parent
    commands = [[str(bin_dir / script), '--help'] for script in args.script] + [[sys.executable, '-m', module, '--help'] for module in args.run_module]
    for command in commands:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f'{" ".join(command)}: {exc}')
            continue
        if done.returncode != 0:
            problems.append(f'{" ".join(command)} exited {done.returncode}: {done.stderr.strip()[-300:]}')
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    content = sub.add_parser('content')
    content.add_argument('dist', type=Path)
    content.add_argument('--package', required=True)
    smoke = sub.add_parser('smoke')
    smoke.add_argument('--distribution', required=True)
    smoke.add_argument('--module', required=True)
    smoke.add_argument('--version')
    smoke.add_argument('--bin-dir')
    smoke.add_argument('--script', action='append', default=[])
    smoke.add_argument('--run-module', action='append', default=[])
    args = parser.parse_args()
    problems = check_content(args.dist, args.package) if args.command == 'content' else check_smoke(args)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
