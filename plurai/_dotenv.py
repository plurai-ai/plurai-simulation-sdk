"""Resolve the ``.env`` that plurai's main-process ``Settings`` should read.

Why this exists
---------------
The agent runs in a subprocess that loads the repo ``.env`` itself (the
customer's ``simulate.py`` seeds ``os.environ``). The *main*
``plurai-run-simulation`` process, by contrast, builds its observability
:class:`~plurai.settings.Settings` from ``os.environ`` ONLY — it never imports
``simulate.py`` and pydantic-settings has no ``env_file`` configured. So nested
``<SINK>__*`` config that lives only in ``.env`` (e.g. ``LANGSMITH__ENABLED``)
is invisible to the main process and the sink silently resolves inactive.

:class:`~plurai.settings.Settings` closes the gap by injecting a pydantic
``DotEnvSettingsSource`` pointed at the path resolved here. That source sits
BELOW the ``os.environ`` source, so a value already exported in the real
environment always wins (pydantic's env > dotenv precedence). ``os.environ`` is
never mutated.

Override / disable
------------------
* ``PLURAI_ENV_FILE=/abs/path/.env`` — use this file instead of searching.
* ``PLURAI_DISABLE_DOTENV=1`` — skip ``.env`` discovery entirely.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv

from plurai.user_path import expand_user_path

_TRUTHY = frozenset({'1', 'true', 'yes', 'on', 't', 'y'})


def resolve_dotenv_path() -> str | None:
    """Return the ``.env`` path ``Settings`` should load, or ``None`` to skip.

    Honors ``PLURAI_DISABLE_DOTENV`` (skip) and ``PLURAI_ENV_FILE`` (explicit
    path, used only when it is a readable file); otherwise walks UP from the cwd
    via python-dotenv's ``find_dotenv`` and returns the nearest match (or
    ``None`` when there is none).
    """
    if os.environ.get('PLURAI_DISABLE_DOTENV', '').strip().lower() in _TRUTHY:
        return None
    override = os.environ.get('PLURAI_ENV_FILE')
    if override:
        expanded = expand_user_path(override)
        return expanded if Path(expanded).is_file() else None
    return find_dotenv(usecwd=True) or None
