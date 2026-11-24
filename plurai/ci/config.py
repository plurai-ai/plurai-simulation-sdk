"""The offline half of ``plurai ci run`` and ``plurai ci validate``: ``ci.yaml``, the committed id files, the env alias and the
SDK pin, read from the simulation directory with no backend and no token.

Every problem is a :class:`Refusal` whose message names what to change; a command exits 65 (``REFUSED``) with it. A message
names files by their path in the repository, never the runner's absolute path, and puts code and paths in backticks. The same
functions run in a pre-commit hook, in ``plurai ci validate`` and as step 1 of ``plurai ci run``, so a configuration that
passes here is not refused by CI for an offline reason.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import yaml

from plurai import __version__
from plurai.child_env import AgentEnvConfigError, declared_env, reserved_agent_env_reason, reserved_agent_env_refusal
from plurai.ci.overrides import OverridesError, Rule, parse_overrides
from plurai.report.outcome import DEFAULT_FAIL_ON_TAGS, REGRESSION_SCENARIO_PREFIX

CI_FILENAME: Final = 'ci.yaml'
SIMULATE_FILENAMES: Final = ('simulate.py', 'simulate.ts')

#: The five keys of ``ci.yaml`` and no others.
KEYS: Final = ('env', 'smoke', 'trials', 'fail_on', 'overrides')
#: The tags that make a scenario × policy gating.
KNOWN_TAGS: Final = ('catastrophic', 'failing', 'standard', 'regression')
#: Scenarios with this prefix are regressions: always selected, and gating through the ``regression`` tag.
REGRESSION_PREFIX: Final = REGRESSION_SCENARIO_PREFIX
DEFAULT_ENV: Final = 'default'
DEFAULT_TRIALS: Final = 3
#: A cell with fewer valid trials is not measured, so fewer trials than this can never pass.
MIN_TRIALS: Final = 2

#: What an id file, an env alias or a scenario name may be: it ends up in a URL path and in a file name.
_ID: Final = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
_SCENARIO: Final = re.compile(r'[^\s/\\]+')
#: The start of the three lines git writes into a file it could not merge.
_CONFLICT_MARKERS: Final = ('<<<<<<<', '=======', '>>>>>>>')
_PACKAGE: Final = 'plurai-simulation-sdk'
_LOCK_FILES: Final = ('uv.lock', 'poetry.lock')
_REQUIREMENT: Final = re.compile(r'^\s*(?:plurai)[-_.]simulation[-_.]sdk\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)', re.IGNORECASE | re.MULTILINE)
_DIST_INFO: Final = re.compile(r'plurai_simulation_sdk-(.+)\.dist-info')


class Refusal(Exception):  # noqa: N818 - a refusal is a result, not an error condition
    """The run cannot start for an offline reason; the message names the files, the values and the fix.

    ``heading`` names the problem in a few plain words, with no values and no markup: it is the heading of the report and the
    title of the check, and the message goes under it.
    """

    def __init__(self, message: str, *, heading: str) -> None:
        super().__init__(message)
        self.heading = heading


@dataclass(frozen=True)
class CiConfig:
    """``ci.yaml`` with its defaults filled in."""

    env: str
    smoke: tuple[str, ...]
    trials: int
    fail_on: tuple[str, ...]
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class SuiteIds:
    """The four committed ids: the suite a run scores."""

    scenarios: str
    evals: str
    env: str
    #: One id per knowledge base, in file order. An agent with no knowledge base commits none.
    corpus: tuple[str, ...]

    @property
    def corpus_text(self) -> str:
        """The corpus ids as ``run.json`` records them: the one id, or all of them sorted and joined with commas; empty for none."""
        return ','.join(sorted(self.corpus))

    def as_run_fields(self) -> dict[str, str]:
        fields = {'scenario_set_id': self.scenarios, 'eval_set_id': self.evals, 'env_id': self.env}
        return {**fields, 'corpus_id': self.corpus_text} if self.corpus else fields


@dataclass(frozen=True)
class Offline:
    """Everything the offline rules read."""

    workdir: Path
    config: CiConfig
    ids: SuiteIds
    #: The ``agent.env`` names the agent declares.
    agent_env: tuple[str, ...]
    simulate: Path


def _read_yaml(path: Path, shown: str) -> Any:  # noqa: ANN401 - whatever the file holds
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise Refusal(f'`{shown}` cannot be read ({error.strerror or error}); fix the file permissions', heading=f'{path.name} cannot be read') from error
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        first = str(error).splitlines()[0] if str(error) else 'parse error'
        raise Refusal(f'`{shown}` is not valid YAML ({first}); fix the syntax', heading=f'{path.name} is not valid YAML') from error


def _names(value: object, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise Refusal(f'ci.yaml: `{key}` is a list of names, not {type(value).__name__}', heading=f'ci.yaml {key} is not valid')
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _SCENARIO.fullmatch(item):
            raise Refusal(
                f'ci.yaml: `{key}` holds {item!r}, which is not a scenario name; list scenario names such as `refund_over_limit`',
                heading=f'ci.yaml {key} is not valid',
            )
        if item in names:
            raise Refusal(f'ci.yaml: `{key}` lists `{item}` twice; list each scenario once', heading=f'ci.yaml {key} lists a scenario twice')
        names.append(item)
    return tuple(names)


def parse_config(document: object) -> CiConfig:
    """The configuration of a parsed ``ci.yaml``. Raises :class:`Refusal` naming the first problem."""
    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise Refusal('ci.yaml is not a mapping; it has the keys ' + ', '.join(f'`{key}`' for key in KEYS), heading='ci.yaml is not a mapping')
    unknown = [str(key) for key in document if key not in KEYS]
    if unknown:
        raise Refusal(f'ci.yaml: `{unknown[0]}` is not a key of ci.yaml; the keys are {", ".join(KEYS)}. Remove it', heading='ci.yaml has an unknown key')
    env = document.get('env', DEFAULT_ENV)
    if not isinstance(env, str) or not _ID.fullmatch(env):
        raise Refusal(f'ci.yaml: `env` is the alias of a directory under `envs/`, such as `default`; got {env!r}', heading='ci.yaml env is not valid')
    trials = document.get('trials', DEFAULT_TRIALS)
    if not isinstance(trials, int) or isinstance(trials, bool):
        raise Refusal(f'ci.yaml: `trials` is a whole number of at least {MIN_TRIALS}; got {trials!r}', heading='ci.yaml trials is not valid')
    if trials < MIN_TRIALS:
        raise Refusal(
            f'ci.yaml: `trials` is {trials}, and a cell needs at least {MIN_TRIALS} valid trials to be measured, so no run could pass; '
            f'set `trials` to {MIN_TRIALS} or more',
            heading=f'ci.yaml trials is below {MIN_TRIALS}',
        )
    fail_on = document.get('fail_on')
    tags = DEFAULT_FAIL_ON_TAGS
    if fail_on is not None:
        if not isinstance(fail_on, Mapping) or set(fail_on) - {'tags'}:
            extra = next((str(k) for k in fail_on if k != 'tags'), None) if isinstance(fail_on, Mapping) else None
            raise Refusal('ci.yaml: `fail_on` has the one key `tags`' + (f'; remove `{extra}`' if extra else ''), heading='ci.yaml fail_on is not valid')
        raw = fail_on.get('tags', list(DEFAULT_FAIL_ON_TAGS))
        if not isinstance(raw, list):
            raise Refusal('ci.yaml: `fail_on.tags` is a list of tags', heading='ci.yaml fail_on.tags is not valid')
        if not raw:
            # An empty list makes every cell advisory, so a run where every cell failed still passes the check.
            raise Refusal(
                'ci.yaml: `fail_on.tags` is empty, so no cell could fail the check; name at least one of '
                f'{", ".join(KNOWN_TAGS)}. To stop a cell from failing the check, add an override with a reason',
                heading='ci.yaml fail_on.tags is empty',
            )
        for tag in raw:
            if tag not in KNOWN_TAGS:
                raise Refusal(
                    f'ci.yaml: `fail_on.tags` names `{tag}`, which is not a tag; the tags are {", ".join(KNOWN_TAGS)}',
                    heading='ci.yaml fail_on.tags names an unknown tag',
                )
        tags = tuple(dict.fromkeys(raw))
    try:
        rules = parse_overrides(document.get('overrides'))
    except OverridesError as error:
        raise Refusal(f'ci.yaml: {error}', heading='ci.yaml overrides are not valid') from error
    return CiConfig(env, _names(document.get('smoke'), 'smoke'), trials, tuple(tags), rules)


def load_config(workdir: Path) -> CiConfig:
    path = workdir / CI_FILENAME
    shown = _shown(workdir, CI_FILENAME)
    if not path.is_file():
        raise Refusal(
            f'`{shown}` does not exist; run /simulation-ci to create it, or add a ci.yaml with the keys {", ".join(KEYS)}', heading='ci.yaml is missing'
        )
    return parse_config(_read_yaml(path, shown))


def _refuse_conflict(text: str, shown: str) -> None:
    """Refuse an id file that holds git conflict markers: regenerating it would drop the id main already has."""
    if any(line.startswith(_CONFLICT_MARKERS) for line in text.splitlines()):
        raise Refusal(
            f"`{shown}` has a merge conflict; resolve it: take main's id and re-apply your edit with the skill that made it",
            heading='an id file has a merge conflict',
        )


def _identifier(value: str, shown: str) -> str:
    if not _ID.fullmatch(value):
        raise Refusal(
            f'`{shown}` does not hold an id (it holds {value[:40]!r}); regenerate it with the skill that wrote it', heading='an id file does not hold an id'
        )
    return value


def parse_corpus_ids(text: str, shown: str) -> tuple[str, ...]:
    _refuse_conflict(text, shown)
    ids = tuple(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    for value in ids:
        if not _ID.fullmatch(value):
            raise Refusal(
                f'`{shown}` holds {value[:40]!r}, which is not an id; regenerate it with /build-rag-corpus', heading='the corpus id file does not hold an id'
            )
    return ids


def _shown(workdir: Path, path: str | Path) -> str:
    """``path`` (relative to ``workdir``, or absolute) as the reader finds it in the repository: relative to the repository
    root, else to the parent of ``workdir`` (the simulation directory sits at the root). A refusal is read on a PR page, where
    the runner's absolute path means nothing."""
    roots = _project_roots(workdir)
    top = roots[-1] if (roots[-1] / '.git').exists() else roots[0].parent
    full = Path(path) if Path(path).is_absolute() else roots[0] / path
    try:
        return full.relative_to(top).as_posix()
    except ValueError:
        return full.as_posix()


@dataclass(frozen=True)
class _SuiteId:
    """A suite id that lives in more than one file: what it names, its marker file and its key in the env manifest."""

    what: str
    marker: str
    key: str


_SCENARIO_SET: Final = _SuiteId('scenario set', '.scenario_set_id', 'scenario_set_id')
_EVAL_SET: Final = _SuiteId('eval set', '.eval_set_id', 'eval_set_id')
_SUITE_IDS: Final = (_SCENARIO_SET, _EVAL_SET)


@dataclass(frozen=True)
class _Copy:
    """One file that holds a suite id: its marker in the env folder or at the root, or its key in the env manifest."""

    where: str
    value: str


def read_ids(workdir: Path, config: CiConfig, *, read: Callable[[str], str | None] | None = None) -> SuiteIds:
    """The four committed ids of the working tree, or of any other source of the same files.

    The env folder ``envs/<env>/`` is where the edit skills write a saved env's ids, so a scenario or eval set id is read from
    the folder's marker, then from the manifest key, and from the root marker only when the folder holds no copy of it.
    ``read(relative_path)`` returns a file's text or ``None`` when it is absent; the default reads ``workdir``. The suite diff
    reads the ids at a git ref through it.
    """
    reader = _reader(workdir, read)
    manifest = _env_manifest(workdir, config.env, reader)
    scenarios = _read_id(_suite_copies(workdir, config.env, manifest, reader, _SCENARIO_SET))
    evals = _read_id(_suite_copies(workdir, config.env, manifest, reader, _EVAL_SET))
    env = _env_id(workdir, config.env, manifest)
    corpus: tuple[str, ...] = ()
    for name in ('.corpus_ids', '.corpus_id'):
        text = reader(name)
        if text is not None:
            corpus = parse_corpus_ids(text, _shown(workdir, name))
            if corpus:
                break
    return SuiteIds(scenarios, evals, env, corpus)


def check_id_copies(workdir: Path, config: CiConfig) -> None:
    """Refuse a scenario or eval set id whose copies (env folder marker, manifest key, root marker) hold different ids.

    An edit writes one place; CI reads another. A disagreement means some copy names a suite the edit left behind, and which
    one the user means is theirs to say, so the refusal names every copy and the id CI would read.
    """
    reader = _reader(workdir, None)
    manifest = _env_manifest(workdir, config.env, reader)
    for suite_id in _SUITE_IDS:
        copies = _suite_copies(workdir, config.env, manifest, reader, suite_id)
        if len({copy.value for copy in copies}) > 1:
            held = ', '.join(f'{copy.where} holds {copy.value}' for copy in copies)
            raise Refusal(
                f'the copies of the {suite_id.what} id disagree: {held}. CI reads {_read_id(copies)} ({copies[0].where}). '
                'Write the id the suite should test into every copy and commit them: run /simulation-ci, '
                'whose readiness check asks which id to keep and writes it into each copy',
                heading=f'the copies of the {suite_id.what} id disagree',
            )


def _read_id(copies: list[_Copy]) -> str:
    """The id CI reads: the first copy, since :func:`_suite_copies` lists them in read order."""
    return copies[0].value


def _manifest_path(alias: str) -> str:
    return f'envs/{alias}/plurai.env.yaml'


def _reader(workdir: Path, read: Callable[[str], str | None] | None) -> Callable[[str], str | None]:
    return read or (lambda name: (workdir / name).read_text(encoding='utf-8') if (workdir / name).is_file() else None)


def _suite_copies(workdir: Path, alias: str, manifest: Mapping[str, Any], reader: Callable[[str], str | None], suite_id: _SuiteId) -> list[_Copy]:
    """Every copy of one suite id, in the order CI reads them. Raises :class:`Refusal` when there is none or one is not an id."""
    what, marker, key = suite_id.what, suite_id.marker, suite_id.key
    env_marker = f'envs/{alias}/{marker}'
    manifest_path = _shown(workdir, _manifest_path(alias))

    def from_marker(name: str) -> list[_Copy]:
        text = reader(name)
        if text is None:
            return []
        value = next((line.strip() for line in text.splitlines() if line.strip()), '')
        shown = _shown(workdir, name)
        _refuse_conflict(text, shown)
        return [_Copy(f'`{shown}`', _identifier(value, shown))]

    def from_manifest() -> list[_Copy]:
        value = manifest.get(key)
        if value is None:
            return []
        if not isinstance(value, str):
            raise Refusal(
                f'`{key}` in `{manifest_path}` is not an id (it holds {value!r}); restore the manifest from git', heading=f'the manifest {key} is not an id'
            )
        return [_Copy(f'`{key}` in `{manifest_path}`', _identifier(value.strip(), manifest_path))]

    copies = [*from_marker(env_marker), *from_manifest(), *from_marker(marker)]
    if not copies:
        env_shown = _shown(workdir, env_marker)
        raise Refusal(
            f'the {what} id is missing: neither `{env_shown}` nor `{key}` in `{manifest_path}` nor `{_shown(workdir, marker)}` '
            f'holds it; commit `{env_shown}` (`{marker}` is written by the generating skill)',
            heading=f'the {what} id is missing',
        )
    return copies


def _env_manifest(workdir: Path, alias: str, reader: Callable[[str], str | None]) -> Mapping[str, Any]:
    relative = _manifest_path(alias)
    text = reader(relative)
    if text is None:
        raise Refusal(
            f'ci.yaml `env: {alias}` has no manifest at `{_shown(workdir, relative)}`; '
            'save the env with /save-env under that alias, or set `env` to an alias under `envs/`',
            heading=f'the env {alias} has no manifest',
        )
    try:
        manifest = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise Refusal(f'`{_shown(workdir, relative)}` is not valid YAML; restore it from git', heading='the env manifest is not valid YAML') from error
    return manifest if isinstance(manifest, Mapping) else {}


def _env_id(workdir: Path, alias: str, manifest: Mapping[str, Any]) -> str:
    env_id = manifest.get('env_id')
    if not isinstance(env_id, str) or not _ID.fullmatch(env_id):
        raise Refusal(
            f'`{_shown(workdir, _manifest_path(alias))}` records no `env_id`; deploy the env with /use-env {alias} and commit the manifest',
            heading='the env manifest records no env_id',
        )
    return env_id


def _agent_env(simulate: Path) -> tuple[str, ...]:
    try:
        names = declared_env(simulate)
    except AgentEnvConfigError as error:
        raise Refusal(f'plurai.config.yaml: {error}', heading='plurai.config.yaml agent.env is not valid') from error
    for name in names:
        if (reason := reserved_agent_env_reason(name)) is not None:
            raise Refusal(
                f'plurai.config.yaml {reserved_agent_env_refusal(name, reason)}; '
                'remove it from agent.env, and give the agent its own provider key under a name of its own',
                heading='agent.env declares a CI credential',
            )
    return names


def _simulate_path(workdir: Path) -> Path:
    for name in SIMULATE_FILENAMES:
        if (workdir / name).is_file():
            return workdir / name
    raise Refusal(
        f'`{_shown(workdir, ".")}` has no simulate.py (or simulate.ts); run `plurai ci` from the simulation directory (`plurai_sim/`)',
        heading='the working directory has no simulate.py',
    )


def _lock_versions(directory: Path) -> Iterable[tuple[Path, str]]:
    import tomllib

    for name in _LOCK_FILES:
        path = directory / name
        if path.is_file():
            try:
                packages = tomllib.loads(path.read_text(encoding='utf-8')).get('package', [])
            except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
                continue
            for package in packages:
                if re.sub(r'[-_.]+', '-', str(package.get('name', ''))).lower() == _PACKAGE and 'version' in package:
                    yield path, str(package['version'])
    for path in sorted(directory.glob('requirements*.txt')):
        for match in _REQUIREMENT.finditer(path.read_text(encoding='utf-8', errors='replace')):
            yield path, match.group(1)


def _project_roots(workdir: Path) -> list[Path]:
    """``workdir`` and each directory above it up to the repository root (the first directory holding ``.git``), else up to
    the filesystem root."""
    roots = []
    for directory in (workdir.resolve(), *workdir.resolve().parents):
        roots.append(directory)
        if (directory / '.git').exists():
            break
    return roots


def check_sdk_pin(workdir: Path, *, version: str = __version__) -> None:
    """Refuse a project whose lockfile or virtual environment holds ``plurai-simulation-sdk`` at another version than this one.

    The project's lock is the source of truth for the SDK, so the SDK that runs and the version the lock names are one version.
    A second version would make the run disagree with the project about the protocol.
    """
    for directory in _project_roots(workdir):
        for path, pinned in _lock_versions(directory):
            if pinned != version:
                raise Refusal(
                    f'`{_shown(workdir, path)}` pins {_PACKAGE} {pinned}, and this run uses {version}; '
                    f'sync the project environment from the lock, or move the lock to {version}',
                    heading='the lock pins another SDK version',
                )
        for site in (*(directory / '.venv' / 'lib').glob('python*/site-packages'), directory / '.venv' / 'Lib' / 'site-packages'):
            for dist in site.glob('plurai_simulation_sdk-*.dist-info'):
                match = _DIST_INFO.fullmatch(dist.name)
                if match and match.group(1) != version:
                    raise Refusal(
                        f'the project environment `{_shown(workdir, directory / ".venv")}` holds {_PACKAGE} {match.group(1)}, and this run uses {version}; '
                        f'sync the project environment from the lock so one version is installed',
                        heading='the project environment holds another SDK version',
                    )


def check_offline(workdir: Path, *, version: str = __version__) -> Offline:
    """All the offline rules, in the order a reader fixes them. Raises :class:`Refusal` naming the first problem."""
    simulate = _simulate_path(workdir)
    config = load_config(workdir)
    ids = read_ids(workdir, config)
    check_id_copies(workdir, config)
    env_names = _agent_env(simulate)
    check_sdk_pin(workdir, version=version)
    return Offline(workdir, config, ids, env_names, simulate)


def today_utc() -> date:
    from datetime import UTC, datetime

    return datetime.now(UTC).date()
