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
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from plurai._endpoints import TOKEN_ENV, URL_ENV
from plurai.host_effects import BACKEND_TOKEN_ENV, CLI_CREDENTIAL_ENV, CLI_SECRET_ENV
from plurai.ingest.workspace_config import CONFIG_FILENAME, WorkspaceConfigError, bench_config_roots, discover_config_documents

#: Top-level ``plurai.config.yaml`` key: the variable names the agent needs from the launching environment.
AGENT_ENV_KEY = 'agent.env'

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


class AgentEnvConfigError(WorkspaceConfigError):
    """``agent.env`` exists but cannot testify to a list of variable names (fail closed)."""


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
    """The names ``document`` declares under :data:`AGENT_ENV_KEY`; empty when it does not declare the key."""
    if AGENT_ENV_KEY not in document:
        return ()
    where = f'`{AGENT_ENV_KEY}` in {source or CONFIG_FILENAME}'
    raw = document[AGENT_ENV_KEY]
    if not isinstance(raw, list):
        raise AgentEnvConfigError(f'{where} must be a list of environment variable names, got {type(raw).__name__}.')
    for name in raw:
        if not isinstance(name, str) or not _VARIABLE_NAME.fullmatch(name):
            raise AgentEnvConfigError(f'{where} lists {name!r}, which is not an environment variable name.')
        if name in BACKEND_TOKEN_ENV:
            raise AgentEnvConfigError(
                f'{where} lists {name}. The backend token reaches the agent only as {TOKEN_ENV}, passed on by the runner; '
                'it is not a variable the agent can ask for.'
            )
    return tuple(dict.fromkeys(raw))


def declared_env(simulate_path: Path | str) -> tuple[str, ...]:
    """The variable names declared beside ``simulate_path`` (its directory and the root above it)."""
    seen: set[Path] = set()
    names: list[str] = []
    for root in bench_config_roots(Path(simulate_path)):
        for path, document in discover_config_documents(root, error=AgentEnvConfigError):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                names.extend(parse_declared_env(document, source=path))
    return tuple(dict.fromkeys(names))


def launch_environment(
    simulate_path: Path | str, *, url: str | None = None, token: str | None = None, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """:func:`child_environment` over this process's environment and the harness's declared names."""
    return child_environment(os.environ, declared=declared_env(simulate_path), url=url, token=token, extra=extra)


__all__ = [
    'AGENT_ENV_KEY',
    'ALLOWLIST_NAMES',
    'ALLOWLIST_PREFIXES',
    'AgentEnvConfigError',
    'child_environment',
    'declared_env',
    'launch_environment',
    'parse_declared_env',
]
