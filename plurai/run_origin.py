"""The run origin: the label a CI run puts on every backend request.

``plurai ci run`` sets ``PLURAI_RUN_ORIGIN=ci`` in its own environment. Every HTTP stack reads it from there: the SDK's httpx
clients (:mod:`plurai._http`) and the mock client and Node shims in the agent's process, which inherit it through the child
allowlist. The backend records the label on threads and jobs so a CI run can be told from a developer's; it is the client's
own claim, gates nothing and is not authentication. A developer's run has no such variable and sends no header.

Stdlib only: the mock client runs inside the customer's agent process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final

#: The environment variable that carries the label, in the runner and in the agent child.
ENV_VAR: Final = 'PLURAI_RUN_ORIGIN'

#: The request header the backend reads.
HEADER: Final = 'X-Plurai-Run-Origin'

#: The one value the backend accepts; any other value is a 400.
CI: Final = 'ci'


def current(env: Mapping[str, str] | None = None) -> str | None:
    """The label to send, ``ci`` when the environment says so and nothing otherwise."""
    return CI if (os.environ if env is None else env).get(ENV_VAR) == CI else None


def mark_ci() -> None:
    """Label this process, and the children it starts, as a CI run."""
    os.environ[ENV_VAR] = CI
