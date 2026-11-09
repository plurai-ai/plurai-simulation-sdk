"""The shape of one row of the login environment table.

A module of its own so that the table (:mod:`plurai.login._environments`) and any extra rows a build carries both import
it and neither imports the other: the table loads the extra rows when a build carries them, and a row module that
imported the table back would make that load order-dependent.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final
from urllib.parse import urlsplit

#: A value the SDK does not carry for a row.
UNASSIGNED: Final = ''


#: The fields a login can do without.
OPTIONAL_FIELDS: Final = frozenset({'billing_url'})


@dataclass(frozen=True)
class Environment:
    name: str
    #: The ``PLURAI_URL`` of this environment, saved in the credentials file; its host selects the row.
    api_root: str
    issuer: str
    #: The base url the key routes hang off: ``<keys_base>/api-keys``.
    keys_base: str
    client_id: str
    #: The page the browser tab is sent to once the key is saved, with a 303. A constant: it carries no account data.
    connected_url: str
    #: The admin page where an organization chooses a package or adds credit, shown in credit-refusal messages. Optional:
    #: a row without one signs in as usual and the messages say "the Plurai dashboard".
    billing_url: str = UNASSIGNED

    @property
    def host(self) -> str:
        return (urlsplit(self.api_root).hostname or '').lower()

    def unassigned(self) -> list[str]:
        """The names of the values this row still lacks; the optional :data:`OPTIONAL_FIELDS` never count."""
        return [item.name for item in fields(self) if item.name not in OPTIONAL_FIELDS and getattr(self, item.name) == UNASSIGNED]
