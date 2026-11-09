"""The httpx clients the SDK opens: ordinary ``httpx`` clients that name the SDK on every request.

``X-Plurai-Client`` is a default header of the client itself, set after the caller's client-level headers, so it reaches
the wire on every request of every call site whatever headers the call site configures. ``User-Agent`` names the SDK the
same way unless the caller already set one. Open clients here and nowhere else;
``tests/test_client_handshake_transport.py`` fails on a client opened any other way.
"""

from __future__ import annotations

from typing import Any

import httpx

from plurai import client_handshake


def _stamped(headers: Any) -> httpx.Headers:  # noqa: ANN401 - any shape httpx takes for ``headers``
    merged = httpx.Headers(headers)
    merged[client_handshake.HEADER] = client_handshake.header_value()
    if client_handshake.USER_AGENT_HEADER not in merged:
        merged[client_handshake.USER_AGENT_HEADER] = client_handshake.user_agent()
    return merged


class Client(httpx.Client):
    def __init__(self, *args: Any, headers: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, headers=_stamped(headers), **kwargs)


class AsyncClient(httpx.AsyncClient):
    def __init__(self, *args: Any, headers: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, headers=_stamped(headers), **kwargs)
