"""The command run id: the label that groups every job one command started on the backend.

The plugin mints one id per command and hands it to the commands it starts, as ``--command-run-id`` or as
``PLURAI_COMMAND_RUN_ID``. A command line here sends it with the requests that start work, and the backend lists the
jobs of one command together. It is a grouping key, not a credential and not a trace id; a run with no id works
the same and is simply not grouped. This package never reads the plugin's files to find it.

Stdlib only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

#: The environment variable a parent process hands the id down in. Part of the client contract.
ENV_VAR: Final = 'PLURAI_COMMAND_RUN_ID'

#: The request header that carries the id.
HEADER: Final = 'X-Plurai-Command-Run-Id'

#: What the backend takes: 1 to 128 visible ASCII characters, no whitespace.
_SHAPE: Final = re.compile(r'[!-~]{1,128}')


def is_valid(value: str) -> bool:
    return _SHAPE.fullmatch(value) is not None


def resolve(flag: str | None, env: Mapping[str, str]) -> str | None:
    """The id this command sends: the flag, else the environment variable, else ``None``.

    Raises:
        ValueError: the value is one the backend would refuse. The message names where it came from.
    """
    if flag is not None and flag.strip():
        source, value = '--command-run-id', flag.strip()
    elif inherited := (env.get(ENV_VAR) or '').strip():
        source, value = ENV_VAR, inherited
    else:
        return None
    if not is_valid(value):
        raise ValueError(f'{source} must be 1 to 128 visible characters with no spaces; got {value[:40]!r}')
    return value
