"""Waiting for the env's mock before the first thread opens.

A request to ``<base>/mock/<env_id>/health`` revives an env the box idle-evicted: the proxy answers 503 ``restarting`` while it
does, and the wait polls until the mock answers 2xx. The states a request cannot fix stop the run: 410 ``needs_redeploy`` (a
person undeployed it), 410 ``unknown_env`` (the backend holds no such env) and the 503 hold of an open edit session. Any other
answer that is not 2xx, and a transport error, is not a ready mock: the wait backs off and tries again until its bound, and the
stop names the last one. A 503 with no code is a sandbox that has never started.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

import httpx

from plurai.ci.backend import Backend, refuse
from plurai.ci.online import StopRun
from plurai.report import Stopped
from plurai.report.run import CAUSES

#: The whole wait, and the polling interval it starts at and backs off to.
BOUND_SECONDS: Final = 300.0
INTERVAL_SECONDS: Final = 3.0
MAX_INTERVAL_SECONDS: Final = 30.0

HOLD_CODE: Final = 'awaiting_edit_finalize'
HOLD_STARTED_KEY: Final = 'edit_session_started_at'


def _code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    code = body.get('code') if isinstance(body, dict) else None
    return code if isinstance(code, str) else None


def _hold_started(response: httpx.Response) -> datetime | None:
    """When the edit session holding the env was opened, from the refusal's ``edit_session_started_at``; ``None`` when the
    backend did not say (an older one) or the value is not a UTC time."""
    try:
        body = response.json()
    except ValueError:
        return None
    raw = body.get(HOLD_STARTED_KEY) if isinstance(body, dict) else None
    try:
        started = datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return started if started is not None and started.tzinfo is not None else None


def _age(seconds: float) -> str:
    """``42 min``, ``3 h`` or ``4 days``: the coarsest unit that keeps the number small."""
    if seconds < 3600:
        return f'{max(1, int(seconds // 60))} min'
    if seconds < 2 * 86400:
        return f'{int(seconds // 3600)} h'
    return f'{int(seconds // 86400)} days'


def _hold_stop(env_id: str, started: datetime | None, now: datetime) -> Stopped:
    """The stop for an env held by an open edit session. The box is shared, so the session blocks every branch's run: the heading
    names the hold and, when the backend said when it started, how long it has stood."""
    fix = 'commit or discard the edit session on the env, then run the job again.'
    if started is None or started > now:
        return Stopped.not_measured('suite', 'env held by an open edit session', fix)
    since = started.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    detail = f'The env {env_id} is held by an edit session open since {since}, which blocks every run on this backend until it ends.'
    return Stopped.not_measured('suite', f'env held by an open edit session for {_age((now - started).total_seconds())}', fix, detail=detail)


def wait_for_mock(
    backend: Backend,
    env_id: str,
    *,
    bound: float = BOUND_SECONDS,
    interval: float = INTERVAL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Returns when the mock answers. Raises :class:`StopRun` for a state a request cannot fix or when ``bound`` seconds
    pass, and the refusal of :func:`~plurai.ci.backend.refuse` for a 401, 403 or 426."""
    route = f'/mock/{env_id}/health'
    deadline = clock() + bound
    wait = interval
    last = 'no answer'
    while True:
        try:
            response: httpx.Response | None = backend.client.get(route)
        except httpx.HTTPError as failure:
            response = None
            last = type(failure).__name__
        if response is not None:
            status = response.status_code
            refuse(response, route)
            code = _code(response)
            if status == 410:
                if code == 'unknown_env':
                    raise StopRun(Stopped.not_on_backend('env', env_id))
                raise StopRun(Stopped.for_cause('needs_redeploy'))
            if status == 503 and code == HOLD_CODE:
                raise StopRun(_hold_stop(env_id, _hold_started(response), wall()))
            if 200 <= status < 300:
                return
            last = f'HTTP {status}'
            # An answer that is not 2xx and carries no code is a sandbox that has never started, a proxy in front of one that
            # is starting, or a route that is not the mock's: back off instead of hammering it.
            wait = min(MAX_INTERVAL_SECONDS, wait * 2) if code is None else interval
        remaining = deadline - clock()
        if remaining <= 0:
            unavailable = CAUSES['unavailable']
            raise StopRun(
                Stopped.not_measured(
                    unavailable.cls,
                    'the mock did not answer in time',
                    f'{unavailable.next_step}.',
                    detail=f'the last answer from {route} was {last}',
                )
            )
        sleep(min(wait, remaining))
