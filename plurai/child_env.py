"""The environment the agent under test starts with.

The child process runs the customer's agent, and a shell-capable agent can be asked by the simulated
user to print its environment. The parent's environment therefore never passes through whole: the child
gets :data:`ALLOWLIST_NAMES` / :data:`ALLOWLIST_PREFIXES` (what a process needs to run, plus the
credential routes the agent's own CLI signs in with), the variables the workspace declares under
``agent.env`` in ``plurai.config.yaml``, and the resolved ``PLURAI_URL`` / ``PLURAI_TOKEN`` pair.

The pair is what the ``code`` action channel needs: the interception hooks of the agent's own shell
dial the hosted mock with that bearer. It is the only route the backend token takes into the child;
a backend token name inherited under any other spelling is dropped (:data:`BACKEND_TOKEN_ENV`), and the
token's value is redacted from everything the SDK writes (:func:`plurai.host_effects.redact_cli_credentials`).

``agent.env`` is a list of variable names, a ROOT key like ``agent.io``. A name that is declared and
absent from the parent environment is simply not set. A malformed declaration, or one that names a
backend token variable, is an error: a declaration that cannot testify never degrades to "nothing
declared".

The root key is the canonical spelling. The same list written nested, ``agent: {env: [...]}``, is read too
(unioned after the root list) and reported (:func:`misplaced_agent_keys`, one warning per config file), so
the run is correct and the config still moves to the root key. A malformed nested list is reported and
ignored, never refused: none of its names reaches the agent.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from plurai import run_origin
from plurai._endpoints import TOKEN_ENV, URL_ENV
from plurai.host_effects import BACKEND_TOKEN_ENV, CLI_CREDENTIAL_ENV, CLI_SECRET_ENV
from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, bench_config_roots, discover_config_documents

logger = logging.getLogger(__name__)

#: Top-level ``plurai.config.yaml`` key: the variable names the agent needs from the launching environment.
AGENT_ENV_KEY = 'agent.env'

#: The ``agent:`` block and the ``env:`` key a nested declaration is written as: ``agent: {env: [...]}``.
_AGENT_KEY = 'agent'
_NESTED_ENV_KEY = 'env'

#: How the nested spelling is named in findings and warnings.
NESTED_AGENT_ENV = 'agent: {env: …}'

#: What a process needs to run: its search paths and interpreter, locale and time zone, home and temp
#: folders, TLS roots and the proxy settings of the network it sits behind (spelled in both cases).
ALLOWLIST_NAMES: frozenset[str] = frozenset(
    {
        'PATH',
        'HOME',
        'USER',
        'LOGNAME',
        'SHELL',
        'TERM',
        'LANG',
        'LANGUAGE',
        'TZ',
        'TMPDIR',
        'TEMP',
        'TMP',
        'VIRTUAL_ENV',
        'CONDA_PREFIX',
        'PYTHONPATH',
        'PYTHONHOME',
        'PYTHONUTF8',
        'PYTHONIOENCODING',
        'PYTHONDONTWRITEBYTECODE',
        'PYTHONUNBUFFERED',
        'NODE_PATH',
        'NODE_EXTRA_CA_CERTS',
        'SSL_CERT_FILE',
        'SSL_CERT_DIR',
        'REQUESTS_CA_BUNDLE',
        'CURL_CA_BUNDLE',
        'HTTP_PROXY',
        'HTTPS_PROXY',
        'ALL_PROXY',
        'NO_PROXY',
        'http_proxy',
        'https_proxy',
        'all_proxy',
        'no_proxy',
        # Windows: a process without these cannot find its system DLLs, profile or temp folders.
        'SYSTEMROOT',
        'SYSTEMDRIVE',
        'WINDIR',
        'COMSPEC',
        'PATHEXT',
        'USERPROFILE',
        'APPDATA',
        'LOCALAPPDATA',
        'PROGRAMDATA',
        'PROGRAMFILES',
        # The provider routes the agent's CLI signs in through (keys, switches, regions). Their values
        # are redacted wherever the SDK writes (:func:`plurai.host_effects.redact_cli_credentials`).
        *CLI_CREDENTIAL_ENV,
        *CLI_SECRET_ENV,
        'AWS_REGION',
        'AWS_DEFAULT_REGION',
        'AWS_PROFILE',
        'CLAUDE_CONFIG_DIR',
    }
    - set(BACKEND_TOKEN_ENV)
)

#: Name families that are the SDK's own settings, locale and XDG folders, and the agent CLI's own knobs.
ALLOWLIST_PREFIXES: tuple[str, ...] = ('PLURAI_', 'NETWORK__', 'LC_', 'XDG_', 'CLAUDE_CODE_', 'ANTHROPIC_')

#: The families of :data:`ALLOWLIST_PREFIXES` that pass on settings only. An operator's own variable can sit under them
#: (``PLURAI_ADMIN_TOKEN``, ``PLURAI_KEY``), so a name in them that reads as a credential is not inherited. The agent CLI's
#: own credential routes are allowlisted by exact name (:data:`ALLOWLIST_NAMES`), and the resolved pair is added separately.
_SETTINGS_PREFIXES: tuple[str, ...] = ('PLURAI_', 'NETWORK__')

#: A name segment that marks a credential: ``PLURAI_ADMIN_TOKEN``, ``NETWORK__DB_PASSWORD``, ``PLURAI_KEY``.
_CREDENTIAL_SEGMENT = re.compile(r'(?:^|_)(?:TOKENS?|KEYS?|SECRETS?|PASSWORDS?|CREDENTIALS?)(?:_|$)')

_VARIABLE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

_GITHUB_COMMAND_FILE = "a GitHub Actions workflow-command file: what the agent writes there changes the job's later steps"
_GITLAB_JOB_JWT = "GitLab's signed job identity token"

#: The names a CI run refuses in ``agent.env``, each with why. A repo declares ``agent.env`` and a pull request can change it,
#: so these are the names that would hand the agent a CI credential, a token endpoint, or a file that steers the job. Each is
#: listed by exact name: a platform's other variables (``CI_ENVIRONMENT``, ``GITHUB_REPOSITORY``, ``RUNNER_TEMP``) carry no
#: credential and are an app's to pass on, unless they read as a credential (:data:`_CI_PLATFORM_PREFIXES`). Matching ignores
#: case. The rule reads names, never values.
RESERVED_AGENT_ENV: Mapping[str, str] = {
    'GITHUB_TOKEN': "the GitHub Actions job's repository token",
    'GH_TOKEN': 'the GitHub CLI token',
    'GH_ENTERPRISE_TOKEN': 'the GitHub CLI token for a GitHub Enterprise host',
    'GITHUB_ENTERPRISE_TOKEN': 'the GitHub CLI token for a GitHub Enterprise host',
    'ACTIONS_ID_TOKEN_REQUEST_TOKEN': 'the GitHub Actions credential that mints OIDC tokens for cloud sign-in',
    'ACTIONS_ID_TOKEN_REQUEST_URL': 'the GitHub Actions endpoint that mints OIDC tokens for cloud sign-in',
    'ACTIONS_RUNTIME_TOKEN': "the GitHub Actions runner's own token",
    'ACTIONS_RUNTIME_URL': "a GitHub Actions runner service endpoint, used with the runner's own token",
    'ACTIONS_RESULTS_URL': "a GitHub Actions runner service endpoint, used with the runner's own token",
    'ACTIONS_CACHE_URL': "a GitHub Actions runner service endpoint, used with the runner's own token",
    'GITHUB_ENV': _GITHUB_COMMAND_FILE,
    'GITHUB_PATH': _GITHUB_COMMAND_FILE,
    'GITHUB_OUTPUT': _GITHUB_COMMAND_FILE,
    'GITHUB_STATE': _GITHUB_COMMAND_FILE,
    'GITHUB_STEP_SUMMARY': _GITHUB_COMMAND_FILE,
    'CI_JOB_TOKEN': "GitLab's job token",
    'CI_BUILD_TOKEN': "GitLab's job token, under its older name",
    'CI_JOB_JWT': _GITLAB_JOB_JWT,
    'CI_JOB_JWT_V1': _GITLAB_JOB_JWT,
    'CI_JOB_JWT_V2': _GITLAB_JOB_JWT,
    'CI_REPOSITORY_URL': "GitLab's clone URL, which carries the job token",
    'CI_REGISTRY_PASSWORD': "GitLab's container registry password",
    'CI_DEPLOY_PASSWORD': "GitLab's deploy token",
    'CI_DEPENDENCY_PROXY_PASSWORD': "GitLab's dependency proxy password",
    'SYSTEM_ACCESSTOKEN': "Azure Pipelines' job token",
    'BUILDKITE_AGENT_ACCESS_TOKEN': "Buildkite's job token",
    'BUILDKITE_AGENT_TOKEN': "Buildkite's agent registration token",
    'CIRCLE_OIDC_TOKEN': "CircleCI's OIDC token",
    'CIRCLE_OIDC_TOKEN_V2': "CircleCI's OIDC token",
    'BITBUCKET_STEP_OIDC_TOKEN': "Bitbucket Pipelines' OIDC token",
    'BUILDKITE_AGENT_JOB_API_SOCKET': "Buildkite's job API socket, which changes the job's environment",
}

#: The prefixes of the CI platforms' own variables. Under them a name that reads as a credential is reserved too, so a token a
#: platform adds later, or one this list misses, is refused rather than passed on.
_CI_PLATFORM_PREFIXES: tuple[str, ...] = (
    'GITHUB_',
    'GH_',
    'ACTIONS_',
    'RUNNER_',
    'CI_',
    'BUILDKITE_',
    'CIRCLE_',
    'BITBUCKET_',
    'TRAVIS_',
    'JENKINS_',
    'SYSTEM_',
)

#: A name segment that marks a CI credential: the :data:`_CREDENTIAL_SEGMENT` words and the token kinds CI platforms name.
_CI_CREDENTIAL_SEGMENT = re.compile(r'(?:^|_)(?:TOKENS?|KEYS?|SECRETS?|PASSWORDS?|CREDENTIALS?|JWT|ACCESSTOKEN|OIDC)(?:_|$)')

_CI_CREDENTIAL_REASON = 'a name under a CI platform prefix that reads as a credential'

#: Plurai's own control: the backend pair, the run origin and the SDK's settings. The runner sets what the agent gets under it
#: (the :data:`ALLOWLIST_PREFIXES` settings that do not read as a credential, and the resolved ``PLURAI_URL`` /
#: ``PLURAI_TOKEN``), so a declaration cannot add to it.
RESERVED_AGENT_ENV_PREFIX = 'PLURAI_'

_PLURAI_REASON = 'a Plurai control variable; the runner sets the ones the agent gets'


def reserved_agent_env_reason(name: str) -> str | None:
    """Why a CI run refuses ``name`` as a declared ``agent.env`` name; ``None`` when it may be declared."""
    upper = name.upper()
    if upper.startswith(RESERVED_AGENT_ENV_PREFIX):
        return _PLURAI_REASON
    if (reason := RESERVED_AGENT_ENV.get(upper)) is not None:
        return reason
    if upper.startswith(_CI_PLATFORM_PREFIXES) and _CI_CREDENTIAL_SEGMENT.search(upper):
        return _CI_CREDENTIAL_REASON
    return None


def reserved_agent_env_refusal(name: str, reason: str) -> str:
    """The one sentence that refuses a reserved ``agent.env`` name, wherever the refusal is raised."""
    return f'`agent.env` declares `{name}`, {reason}. Such names never reach the agent through `agent.env`'


class AgentEnvConfigError(WorkspaceConfigError):
    """``agent.env`` exists but cannot testify to a list of variable names (fail closed)."""


@dataclass(frozen=True, slots=True)
class MisplacedAgentKey:
    """A nested ``agent: {env: …}`` in one config file: read as :data:`AGENT_ENV_KEY`, or ignored as malformed."""

    #: The spelling found (:data:`NESTED_AGENT_ENV`).
    key: str
    #: The key it was read as; ``None`` when it was ignored.
    read_as: str | None
    #: Why it was ignored; ``None`` when it was read.
    defect: str | None
    source: Path


def _env_list_defect(raw: object) -> str | None:
    """Why ``raw`` is not a list of variable names the agent may ask for; ``None`` when it is."""
    if not isinstance(raw, list):
        return f'must be a list of environment variable names, got {type(raw).__name__}.'
    for name in raw:
        if not isinstance(name, str) or not _VARIABLE_NAME.fullmatch(name):
            return f'lists {name!r}, which is not an environment variable name.'
        if name in BACKEND_TOKEN_ENV:
            return (
                f'lists {name}. The backend token reaches the agent only as {TOKEN_ENV}, passed on by the runner; it is not a variable the agent can ask for.'
            )
    return None


def _nested_env(document: Mapping[str, object]) -> tuple[tuple[str, ...], str | None] | None:
    """The nested ``agent: {env: …}`` list as ``(names, defect)``; ``None`` when ``document`` has none.

    THE one reader of the nested spelling, so the names the child gets and the defect the run reports never
    disagree. A malformed list yields no names at all, not its well-formed part.
    """
    agent = document.get(_AGENT_KEY)
    if not isinstance(agent, Mapping) or _NESTED_ENV_KEY not in agent:
        return None
    raw = agent[_NESTED_ENV_KEY]
    if (defect := _env_list_defect(raw)) is not None:
        return (), defect
    return tuple(dict.fromkeys(cast('list[str]', raw))), None


def _inherited(name: str) -> bool:
    if name in BACKEND_TOKEN_ENV:
        return False
    if name in ALLOWLIST_NAMES:
        return True
    return name.startswith(ALLOWLIST_PREFIXES) and not (name.startswith(_SETTINGS_PREFIXES) and _CREDENTIAL_SEGMENT.search(name))


def child_environment(
    parent: Mapping[str, str],
    *,
    declared: Iterable[str] = (),
    url: str | None = None,
    token: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The child's environment: ``parent`` filtered to the allowlist and ``declared`` names, then
    ``extra``, then the pair.

    ``url`` / ``token`` are the resolved pair. Without a ``url`` the pair is the parent's own ``PLURAI_URL`` /
    ``PLURAI_TOKEN``; with one, ``token`` is used as given, because a token the parent holds belongs to whichever
    backend the parent names. A blank value is no value.
    """
    names = set(declared)
    if run_origin.current(parent) == run_origin.CI:
        for name in sorted(names):
            if (reason := reserved_agent_env_reason(name)) is not None:
                raise AgentEnvConfigError(f'{reserved_agent_env_refusal(name, reason)}; remove it from agent.env.')
    env = {name: value for name, value in parent.items() if _inherited(name) or name in names}
    env.update(extra or {})
    resolved_url = url or parent.get(URL_ENV)
    resolved_token = token if url else (token or parent.get(TOKEN_ENV))
    for name, value in ((URL_ENV, resolved_url), (TOKEN_ENV, resolved_token)):
        if value and value.strip():
            env[name] = value
        else:
            env.pop(name, None)
    return env


def parse_declared_env(document: Mapping[str, object], *, source: Path | None = None) -> tuple[str, ...]:
    """The names ``document`` declares: :data:`AGENT_ENV_KEY`, then a well-formed nested ``agent: {env: …}``.

    A malformed root key raises :class:`AgentEnvConfigError`; a malformed nested list adds nothing (it is
    reported by :func:`misplaced_agent_keys`). Empty when neither is declared.
    """
    names: tuple[str, ...] = ()
    if AGENT_ENV_KEY in document:
        raw = document[AGENT_ENV_KEY]
        if (defect := _env_list_defect(raw)) is not None:
            raise AgentEnvConfigError(f'`{AGENT_ENV_KEY}` in {source or CONFIG_FILENAME} {defect}')
        names = tuple(cast('list[str]', raw))
    if (nested := _nested_env(document)) is not None:
        names += nested[0]
    return tuple(dict.fromkeys(names))


def _config_documents(simulate_path: Path | str) -> Iterator[tuple[Path, Mapping[str, object]]]:
    """Each config document beside ``simulate_path`` (its directory and the root above it), once per file."""
    seen: set[Path] = set()
    for root in bench_config_roots(Path(simulate_path)):
        for path, document in discover_config_documents(root, error=AgentEnvConfigError):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path, document


#: Config files whose nested ``agent: {env: …}`` this process already warned about.
_WARNED_NESTED: set[Path] = set()


def _warn_nested_once(path: Path, nested: tuple[tuple[str, ...], str | None]) -> None:
    resolved = path.resolve()
    if resolved in _WARNED_NESTED:
        return
    _WARNED_NESTED.add(resolved)
    names, defect = nested
    if defect is None:
        logger.warning(
            '%s: `%s` is read as the root key `%s` (%s). Move the list to the root of %s, written `%s:` beside `agent.io:`.',
            path,
            NESTED_AGENT_ENV,
            AGENT_ENV_KEY,
            ', '.join(names) or 'no names',
            CONFIG_FILENAME,
            AGENT_ENV_KEY,
        )
    else:
        logger.warning(
            '%s: `%s` %s None of its names reaches the agent. Write it as the root key `%s: [NAME, …]`.',
            path,
            NESTED_AGENT_ENV,
            defect,
            AGENT_ENV_KEY,
        )


def declared_env(simulate_path: Path | str) -> tuple[str, ...]:
    """The variable names declared beside ``simulate_path`` (its directory and the root above it).

    A nested ``agent: {env: …}`` is read with the root key and warned about once per config file.
    """
    names: list[str] = []
    for path, document in _config_documents(simulate_path):
        names.extend(parse_declared_env(document, source=path))
        if (nested := _nested_env(document)) is not None:
            _warn_nested_once(path, nested)
    return tuple(dict.fromkeys(names))


def misplaced_agent_keys(simulate_path: Path | str) -> tuple[MisplacedAgentKey, ...]:
    """Each nested ``agent: {env: …}`` beside ``simulate_path``: read as :data:`AGENT_ENV_KEY`, or ignored as malformed.

    Raises :class:`AgentEnvConfigError` when a config file cannot be read at all, as :func:`declared_env` does.
    """
    out: list[MisplacedAgentKey] = []
    for path, document in _config_documents(simulate_path):
        if (nested := _nested_env(document)) is not None:
            _, defect = nested
            out.append(MisplacedAgentKey(key=NESTED_AGENT_ENV, read_as=None if defect else AGENT_ENV_KEY, defect=defect, source=path))
    return tuple(out)


def launch_environment(
    simulate_path: Path | str, *, url: str | None = None, token: str | None = None, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """:func:`child_environment` over this process's environment and the harness's declared names."""
    return child_environment(os.environ, declared=declared_env(simulate_path), url=url, token=token, extra=extra)


__all__ = [
    'AGENT_ENV_KEY',
    'ALLOWLIST_NAMES',
    'ALLOWLIST_PREFIXES',
    'NESTED_AGENT_ENV',
    'AgentEnvConfigError',
    'MisplacedAgentKey',
    'child_environment',
    'declared_env',
    'reserved_agent_env_reason',
    'reserved_agent_env_refusal',
    'launch_environment',
    'misplaced_agent_keys',
    'parse_declared_env',
]
