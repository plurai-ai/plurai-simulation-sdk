"""The pages the loopback listener of ``plurai login`` shows the browser tab.

There are two, both constant documents in ``plurai/login/pages/``: the sign-in failed, and not found (any request that
is not the one expected callback). A successful sign-in sends the browser to the environment's own page instead. They
are plain HTML files so a designer can open and replace them; ``package-data`` in ``pyproject.toml`` ships them in the
wheel. A request can never put its own text on a page, so no query value, ``error_description``, token, code, ``state``
or account name reaches the output, and the page says nothing of why a sign-in failed: the terminal carries the reason.
The pages are self-contained: inline CSS, an inline SVG logo and icon, nothing fetched from outside, no script, no claim
that the tab closes by itself. The listener sends them with a content security policy that allows nothing else
(:data:`SECURITY_HEADERS`).

Stdlib only.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final, Literal

State = Literal['failed', 'not_found']

#: Sent with every answer, the redirect included. ``default-src 'none'`` plus inline styles and data images is all the
#: pages use; framing is refused.
SECURITY_HEADERS: Final = {
    'Content-Security-Policy': "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'",
    'X-Content-Type-Options': 'nosniff',
    'Cache-Control': 'no-store',
    'Referrer-Policy': 'no-referrer',
}

_STATES: Final[tuple[State, ...]] = ('failed', 'not_found')

#: The body of every state, read once at import: a page that is missing from the install fails the import, not a sign-in.
PAGES: Final[dict[str, bytes]] = {state: files('plurai.login').joinpath('pages', f'{state}.html').read_bytes() for state in _STATES}


def page(state: State) -> bytes:
    """The body for ``state``."""
    return PAGES[state]
