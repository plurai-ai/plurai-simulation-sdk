"""The backends ``plurai login`` can sign in to, keyed by the host of their api root (``PLURAI_URL``).

Each row names what the browser login needs and the backend does not advertise: the authorization server (issuer), the
base url of the backend's key routes, the client id, which is the url of the client metadata document the backend serves
for this client, and the page of the Plurai site the browser tab is sent to once the key is saved. The SDK carries the
table because the values are stable per environment; there is no discovery document on the api root and no variable that
overrides a row. Any other host is refused, so a login never sends a sign-in to a server the SDK does not know.

A row holding :data:`UNASSIGNED` in any value cannot be used for a login; the refusal names no missing value, because the
values are the SDK's own wiring and the user can do nothing about them. The SDK has no success page of its own, so an
environment goes live only with its connected page.

The table holds production first. A build may carry extra rows after it; a build without them knows production alone and
refuses any other host.

Stdlib only.
"""

from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit

from plurai.login._row import UNASSIGNED, Environment

__all__ = ['ENVIRONMENTS', 'PRODUCTION', 'UNASSIGNED', 'Environment', 'billing_url_for', 'default', 'for_host', 'known_hosts']

PRODUCTION: Final = Environment(
    name='production',
    api_root='https://sandbox.plurai.ai/v1',
    issuer='https://clerk.simulation.plurai.ai',
    keys_base='https://api.plurai.ai/simulation/v1',
    client_id='https://typhoon.plurai.ai/oauth/clients/sdk/metadata.json',  # leak-allow: codename -- public URL the sign-in server fetches
    connected_url='https://simulation.plurai.ai/connect?step=run&connected=1',
    billing_url='https://simulation.plurai.ai/packages',
)


def _development_rows() -> tuple[Environment, ...]:
    """The extra rows this build carries; none in a build without them."""
    try:
        from plurai.login._dev_environments import ROWS  # noqa: PLC0415 - the module is not part of every build
    except ModuleNotFoundError as missing:
        # Only the module itself being absent means a build without it; a dependency of it failing to import is a bug to show.
        if missing.name != 'plurai.login._dev_environments':
            raise
        return ()
    return ROWS


#: Production first: it is the backend of a login with ``PLURAI_URL`` unset.
ENVIRONMENTS: tuple[Environment, ...] = (PRODUCTION, *_development_rows())


def default() -> Environment:
    return ENVIRONMENTS[0]


def for_host(host: str) -> Environment | None:
    wanted = host.lower()
    return next((row for row in ENVIRONMENTS if row.api_root != UNASSIGNED and row.host == wanted), None)


def billing_url_for(api_url: str | None) -> str:
    """The billing page of the environment whose api root has the host of *api_url*; empty when there is none or the host is unknown."""
    try:
        host = urlsplit(api_url or '').hostname or ''
    except ValueError:
        return ''
    row = for_host(host) if host else None
    return row.billing_url if row is not None else UNASSIGNED


def known_hosts() -> str:
    """The environments and their hosts, for a refusal: ``<name> (<host>)`` for each row, comma separated."""
    return ', '.join(f'{row.name} ({row.host or "not assigned yet"})' for row in ENVIRONMENTS)
