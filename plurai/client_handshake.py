"""The client-version handshake: how the SDK names itself to the backend and what it says when the backend refuses it.

Every request to a Plurai backend carries ``X-Plurai-Client: plurai-simulation-sdk/<version>`` (and the same token as its
``User-Agent``, unless the caller set one), stamped by the
transport (:mod:`plurai._http`, and the mock client's opener) so no call site and no skill text can leave it out.
The backend refuses a client it cannot identify, or one older than its floor, with HTTP 426 and a typed
``client_upgrade_required`` body. :func:`upgrade_message` turns that refusal into the one ``pip`` line that fixes it;
the command exits :data:`plurai.exit_codes.UPGRADE_REQUIRED`. Under the Plurai CI action (see :data:`ACTION_ENV`) the
``pip`` line is the wrong fix, because the action installs the SDK, so the text asks for a newer action release instead.

Only the floor, validated as a version, reaches the terminal: the backend's own prose is never echoed, and a floor
that is not a version is dropped, because the line is meant to be copied into a shell.

Stdlib only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from typing import Any, Final, Protocol

#: The name the backend keeps this client's floor under; it is also the PyPI distribution name.
CLIENT_NAME: Final = 'plurai-simulation-sdk'

#: The request header naming the caller: ``<client>/<version>``.
HEADER: Final = 'X-Plurai-Client'

#: The response header a Plurai backend adds to every answer to an identified client: ``<client>/<floor>``. Its absence
#: on a refusal is one of the facts :mod:`plurai.network_refusal` reads to say the refusal did not come from Plurai.
FLOOR_HEADER: Final = 'X-Plurai-Min-Client'

#: The ordinary agent header. Some HTTP front ends refuse Python's default urllib ``Python-urllib/*`` agent, so every
#: request names the SDK here as well; removing the header makes those front ends reject the request.
USER_AGENT_HEADER: Final = 'User-Agent'

#: Set by the Plurai CI action, to any non-blank value, in the environment of the SDK it installs. Its presence is the
#: only fact the SDK reads from it.
ACTION_ENV: Final = 'PLURAI_ACTION_VERSION'

#: The action's repository and the page that lists its release tags; the SDK cannot know the newest tag, so it names the page.
ACTION_REPOSITORY: Final = 'plurai-ai/simulate-action'
ACTION_RELEASES_URL: Final = f'https://github.com/{ACTION_REPOSITORY}/releases'

#: The HTTP status of a refusal.
UPGRADE_STATUS: Final = 426

#: The ``error`` of a refusal body.
REFUSAL_ERROR: Final = 'client_upgrade_required'

#: The closed vocabulary of ``reason``; an unrecognised one is still a refusal.
REASONS: Final = frozenset({'missing_header', 'malformed_header', 'unknown_client', 'below_floor'})

#: The backend's version grammar: ``X.Y.Z`` plus an optional PEP 440 pre or post suffix. A header value outside it is
#: refused as malformed, so the version sent is held to it (see :func:`version`); a floor outside it is never printed.
_VERSION: Final = re.compile(r'(\d{1,9})\.(\d{1,9})\.(\d{1,9})(?:[-.]?(?:dev|a|b|rc)\.?\d{1,9}|\.post\d{1,9})?', re.ASCII)


@cache
def version() -> str:
    """The installed SDK version, without a PEP 440 local segment (``+g1a2b3c``), which the backend does not accept."""
    from plurai import __version__  # noqa: PLC0415 - the package imports this module's callers; a lazy read keeps the cycle closed

    return __version__.partition('+')[0]


def header_value() -> str:
    return f'{CLIENT_NAME}/{version()}'


def user_agent() -> str:
    """The ``User-Agent`` every request carries: the same ``<client>/<version>`` token as :data:`HEADER`."""
    return header_value()


def headers() -> dict[str, str]:
    return {HEADER: header_value()}


@dataclass(frozen=True)
class UpgradeRefusal:
    """A 426 from the backend, reduced to what the user acts on."""

    #: One of :data:`REASONS`, or ``None`` for a reason this client does not know or a body that carries none.
    reason: str | None
    #: The lowest SDK version the backend serves, or ``None`` when the answer did not carry a usable one.
    floor: str | None


class _Response(Protocol):
    status_code: int

    def json(self) -> Any: ...  # noqa: ANN401 - whatever JSON the backend answered


def _as_version(value: object) -> str | None:
    return value if isinstance(value, str) and _VERSION.fullmatch(value) else None


def _floor(body: dict[str, Any]) -> str | None:
    """The SDK's floor: the refusal's own ``minimum_version`` when it concerns this client, else the published map's entry."""
    if body.get('client') == CLIENT_NAME and (floor := _as_version(body.get('minimum_version'))) is not None:
        return floor
    floors = body.get('minimum_versions')
    return _as_version(floors.get(CLIENT_NAME)) if isinstance(floors, dict) else None


def refusal_from(status: int, body: object) -> UpgradeRefusal | None:
    """The refusal a ``status`` and parsed JSON ``body`` make, or ``None`` when the answer is not a 426.

    A 426 whose body is not the typed refusal (a gateway's own, say) is still a refusal, with no floor.
    """
    if status != UPGRADE_STATUS:
        return None
    if not isinstance(body, dict) or body.get('error') != REFUSAL_ERROR:
        return UpgradeRefusal(reason=None, floor=None)
    reason = body.get('reason')
    return UpgradeRefusal(reason=reason if isinstance(reason, str) and reason in REASONS else None, floor=_floor(body))


def refusal_from_response(response: _Response) -> UpgradeRefusal | None:
    """:func:`refusal_from` for a response; a body that is not JSON carries no floor."""
    try:
        body = response.json()
    except ValueError:
        body = None
    return refusal_from(response.status_code, body)


def pip_line(floor: str | None) -> str:
    return f"pip install -U '{CLIENT_NAME}>={floor}'" if floor else f"pip install -U '{CLIENT_NAME}'"


def _needed_version_clause(refusal: UpgradeRefusal) -> str:
    """The clause that says which version Plurai accepts, from the floor when the refusal carries one."""
    return f'it needs {refusal.floor} or later' if refusal.floor else 'it needs a newer release'


def _under_action() -> bool:
    """Whether this SDK runs under the Plurai CI action."""
    return bool(os.environ.get(ACTION_ENV, '').strip())


def _action_upgrade_message(refusal: UpgradeRefusal) -> str:
    """Two lines for a refused SDK under the action: which release is too old, then the workflow line to change."""
    return (
        f'This release of {ACTION_REPOSITORY} installs {CLIENT_NAME} {version()}, which Plurai no longer accepts '
        f'({_needed_version_clause(refusal)}).\n'
        f"Move the workflow's uses: {ACTION_REPOSITORY}@<ref> to the newest release tag: {ACTION_RELEASES_URL}"
    )


def upgrade_message(refusal: UpgradeRefusal) -> str:
    """What the backend refused, then the ``pip`` command alone on the last line.

    Under the action no text carries a ``pip`` line, because the action installs the SDK: a refusal about the SDK's
    version asks for a newer action release, and any other refusal keeps its first line alone.
    """
    if refusal.reason in (None, 'below_floor'):
        if _under_action():
            return _action_upgrade_message(refusal)
        first = f'the backend refuses this {CLIENT_NAME} ({version()}): {_needed_version_clause(refusal)}. Upgrade it:'
    else:
        first = (
            f'the backend could not accept the {HEADER} header this {CLIENT_NAME} ({version()}) sends '
            f'({refusal.reason.replace("_", " ")}); a proxy in front of it may strip the header.'
        )
        if _under_action():
            return first
        first += ' Upgrade, then retry:'
    return f'{first}\n{pip_line(refusal.floor)}'


def upgrade_message_for(response: _Response) -> str:
    """The message for a 426 ``response``."""
    return upgrade_message(refusal_from_response(response) or UpgradeRefusal(reason=None, floor=None))


class UpgradeRequiredError(Exception):
    """Raised by a check that runs before a command's own error handling; the message is :func:`upgrade_message`."""
