"""The backend reads a CI run makes before and around its scenarios: artifacts, the credit state, the key list and the mock.

A 401 on any call is :class:`TokenRejected`, a 403 is :class:`TokenRefused` and a 426 is
:class:`UpgradeRefused`; none is retried. A transport failure or a
5xx that outlives the retries is :class:`BackendUnreachable`. While :attr:`Backend.resume_seconds` is above zero a 503 that
outlives the retries is read again, with a line on stderr, until that time is spent: the box is resuming from a stop. Requests
go through :mod:`plurai._http`, so they carry ``X-Plurai-Client`` and, in a CI run, ``X-Plurai-Run-Origin``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date
from typing import Any, Final

import httpx

from plurai import _http, client_handshake
from plurai.ci.isotime import parse_instant
from plurai._http_retry import backoff_seconds, parse_retry_after, throttled_wait

#: Statuses a read is repeated on: the box restarting, a proxy timing out, a throttle.
_RETRIED: Final = frozenset({429, 502, 503, 504})
_ATTEMPTS: Final = 3
_TIMEOUT_SECONDS: Final = 30.0
#: How long the first reads of a run wait for a backend box that is being stopped or woken: it answers 503 until it is up.
RESUME_SECONDS: Final = 120.0
#: The wait between two reads while the box resumes.
_RESUME_STEP_SECONDS: Final = 5.0


class BackendError(Exception):
    """A read failed."""


class TokenRejected(BackendError):  # noqa: N818
    """The backend answered 401: the token is expired, revoked or wrong."""

    def __init__(self, route: str) -> None:
        super().__init__(f'token rejected on {route}')
        self.route = route


class TokenRefused(BackendError):  # noqa: N818
    """The backend answered 403: the token is valid and may not call this route."""

    def __init__(self, route: str) -> None:
        super().__init__(f'token refused for {route}')
        self.route = route


class UpgradeRefused(BackendError):  # noqa: N818
    """The backend answered 426: it refuses this SDK version, on every call, until the SDK is newer."""

    def __init__(self, route: str, refusal: client_handshake.UpgradeRefusal) -> None:
        super().__init__(f'upgrade required on {route}')
        self.route = route
        self.refusal = refusal


class BackendUnreachable(BackendError):  # noqa: N818
    """The backend did not answer, or answered something this client cannot use, after the retries."""


#: The refusals every call can meet: none is retried and none is a problem of the one call, so a caller that handles
#: another failure of its call lets these through.
REFUSALS = (TokenRejected, TokenRefused, UpgradeRefused)


def refuse(response: httpx.Response, route: str) -> None:
    """Raise the refusal a 401, 403 or 426 answer is; return for any other answer."""
    if response.status_code == 401:
        raise TokenRejected(route)
    if response.status_code == 403:
        raise TokenRefused(route)
    if (refusal := client_handshake.refusal_from_response(response)) is not None:
        raise UpgradeRefused(route, refusal)


@dataclass(frozen=True)
class Link:
    kind: str
    id: str
    role: str


@dataclass(frozen=True)
class CorpusDocument:
    """One source document of a published corpus: its name and a digest of its chunk texts."""

    name: str
    sha256: str


@dataclass(frozen=True)
class Artifact:
    """``GET /artifacts/{id}`` for an id the backend holds."""

    id: str
    kind: str
    status: str
    links: tuple[Link, ...] = ()
    #: The id this artifact was copied from at a copy-on-write release.
    lineage: str | None = None
    pinned_digest: str | None = None
    pinned_digest_drift: bool = False
    posture: str | None = None
    #: A scenario set's scenarios as ``(name, solvability)``.
    scenarios: tuple[tuple[str, str], ...] = ()
    #: An eval set's policies as ``(name, effective_tier)``; ``None`` when the backend could not load them.
    policy_tiers: tuple[tuple[str, str], ...] | None = None
    #: A corpus's documents sorted by name; empty for a corpus with none, ``None`` when the backend lists none (it could not,
    #: or the artifact is not a published corpus) or lists rows this client cannot read in full.
    documents: tuple[CorpusDocument, ...] | None = None

    @property
    def unsolvable(self) -> tuple[str, ...]:
        return tuple(name for name, state in self.scenarios if state == 'unsolvable')


@dataclass(frozen=True)
class CreditState:
    """``GET /credits``: ``ok``, or the gateway's refusal code and when it resets."""

    state: str
    resets_at: str | None = None


@dataclass(frozen=True)
class KeyExpiry:
    """What the key list says about the key named after the repo."""

    expires: date | None
    note: str | None = None


@dataclass
class Backend:
    """A blocking client for the CI reads. ``sleep`` and ``client`` are injectable."""

    url: str
    token: str
    client: httpx.Client = field(default=None)  # type: ignore[assignment]
    sleep: Callable[[float], None] = time.sleep
    #: The seconds a 503 that outlived the retries is still read again for, spent across calls: the box resuming from a stop. Zero
    #: ends a 503 with the ordinary retries.
    resume_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.url = self.url.rstrip('/')
        if self.client is None:
            self.client = _http.Client(base_url=self.url, headers={'Authorization': f'Bearer {self.token}'}, timeout=_TIMEOUT_SECONDS, follow_redirects=False)

    def close(self) -> None:
        self.client.close()

    def request(self, method: str, path: str, *, route: str | None = None) -> httpx.Response:
        """The response to ``method path``. Raises :class:`TokenRejected`, :class:`TokenRefused`,
        :class:`UpgradeRefused` or :class:`BackendUnreachable`."""
        label = route or path
        response: httpx.Response | None = None
        failure: Exception | None = None
        attempt = 0
        announced = False
        while True:
            attempt += 1
            try:
                response = self.client.request(method, path)
                failure = None
            except httpx.HTTPError as error:
                response, failure = None, error
            if response is not None:
                refuse(response, label)
                if response.status_code not in _RETRIED:
                    return response
            if attempt < _ATTEMPTS:
                after = parse_retry_after(response.headers.get('Retry-After')) if response is not None else None
                self.sleep(throttled_wait(backoff_seconds(attempt), after, cap=10.0))
                continue
            if response is None or response.status_code != 503 or self.resume_seconds <= 0:
                break
            if not announced:
                print(
                    f'plurai ci run: {label} answered 503: the backend may be waking up, waiting for it to resume (up to {self.resume_seconds:.0f} s)',
                    file=sys.stderr,
                )
                announced = True
            step = min(_RESUME_STEP_SECONDS, self.resume_seconds)
            self.resume_seconds -= step
            self.sleep(step)
        if response is not None:
            raise BackendUnreachable(f'{label} answered HTTP {response.status_code}')
        raise BackendUnreachable(f'{label} did not answer ({type(failure).__name__})')

    def get(self, path: str, *, route: str) -> httpx.Response:
        """``GET path`` with the token rules and retries of :meth:`request`."""
        return self.request('GET', path, route=route)

    def artifact(self, artifact_id: str) -> Artifact | None:
        """The artifact, or ``None`` when the backend holds no such id."""
        response = self.request('GET', f'/artifacts/{artifact_id}', route='GET /artifacts/{id}')
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise BackendUnreachable(f'GET /artifacts/{{id}} answered HTTP {response.status_code}')
        return _artifact(_json(response, 'GET /artifacts/{id}'))

    def credits(self) -> CreditState | None:
        """The org's credit state, or ``None`` when the backend exposes none: the route is absent, or its answer names no state."""
        response = self.request('GET', '/credits')
        if response.status_code in (404, 405, 501):
            return None
        if response.status_code != 200:
            raise BackendUnreachable(f'GET /credits answered HTTP {response.status_code}')
        body = _json(response, 'GET /credits')
        state = body.get('state')
        resets = body.get('resets_at')
        if not isinstance(state, str):
            # An answer that does not name a state says nothing about the credits: it is no "ok".
            return None
        return CreditState(state, resets if isinstance(resets, str) else None)

    def key_expiry(self, repo: str) -> KeyExpiry:
        """The expiry of the service-account key named ``repo``, the name ``plurai ci token`` gives it. A list that fails, or
        has no such key, is a note and never a stop.

        ``GET /api-keys`` answers a JSON array of keys, each with ``name``, ``identity.type`` and ``expiresAt`` (an ISO-8601
        instant, or ``null`` for a key that never expires); it leaves out revoked and expired keys, and a name is unique among
        the rest. A 401 still means the token is rejected. A 403 on this route only means the key may not list keys.
        """
        unknown = KeyExpiry(None, 'token expiry unknown')
        # The call only adds a note, so it never spends the window the reads that decide the run wait in.
        window, self.resume_seconds = self.resume_seconds, 0.0
        try:
            response = self.request('GET', '/api-keys')
        except TokenRefused:
            return unknown
        except BackendUnreachable:
            return unknown
        finally:
            self.resume_seconds = window
        if response.status_code != 200:
            return unknown
        try:
            body = response.json()
        except ValueError:
            return unknown
        for row in body if isinstance(body, list) else ():
            if isinstance(row, Mapping) and row.get('name') == repo and _service_key(row):
                if row.get('expiresAt', '') is None:
                    return KeyExpiry(None)
                expires = parse_instant(row.get('expiresAt'))
                return KeyExpiry(expires.astimezone(UTC).date()) if expires else unknown
        return unknown


def _service_key(row: Mapping[str, Any]) -> bool:
    identity = row.get('identity')
    return isinstance(identity, Mapping) and identity.get('type') == 'serviceAccount'


def _json(response: httpx.Response, route: str) -> Mapping[str, Any]:
    try:
        body = response.json()
    except ValueError as error:
        raise BackendUnreachable(f'{route} did not answer JSON') from error
    if not isinstance(body, Mapping):
        raise BackendUnreachable(f'{route} did not answer a JSON object')
    return body


def _documents(raw: object) -> tuple[CorpusDocument, ...] | None:
    """The listing, or ``None`` for anything but a list of complete rows: a partial list would invent additions and removals."""
    if not isinstance(raw, list):
        return None
    documents: list[CorpusDocument] = []
    for row in raw:
        if not (isinstance(row, Mapping) and isinstance(row.get('name'), str) and isinstance(row.get('sha256'), str)):
            return None
        documents.append(CorpusDocument(row['name'], row['sha256']))
    return tuple(documents)


def _policy_tiers(raw: object) -> tuple[tuple[str, str], ...] | None:
    """``(policy, tier)`` of every row, or ``None`` when the read cannot be trusted: not a list, or a row with no name or no string
    tier. A dropped row would leave its policy with no cell, and a null or missing tier would make it advisory, so the run
    must not go on with a partial list."""
    if not isinstance(raw, list):
        return None
    rows: list[tuple[str, str]] = []
    for row in raw:
        if not (isinstance(row, Mapping) and isinstance(row.get('name'), str) and isinstance(row.get('effective_tier'), str)):
            return None
        rows.append((row['name'], row['effective_tier']))
    return tuple(rows)


def _artifact(body: Mapping[str, Any]) -> Artifact:
    raw = body.get('detail')
    detail: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    links = tuple(
        Link(str(row['kind']), str(row['id']), str(row.get('role', '')))
        for row in detail.get('links') or []
        if isinstance(row, Mapping) and 'id' in row and 'kind' in row
    )
    scenarios = tuple(
        (str(row['name']), str(row.get('solvability', 'untested'))) for row in detail.get('scenarios') or [] if isinstance(row, Mapping) and 'name' in row
    )
    policy_tiers = _policy_tiers(detail.get('policy_tiers'))
    lineage = detail.get('lineage')
    digest = detail.get('pinned_digest')
    return Artifact(
        id=str(body.get('id', '')),
        kind=str(body.get('kind', '')),
        status=str(body.get('status', '')),
        links=links,
        lineage=lineage if isinstance(lineage, str) and lineage else None,
        pinned_digest=digest if isinstance(digest, str) and digest else None,
        pinned_digest_drift=detail.get('pinned_digest_drift') is True,
        posture=detail.get('ingestion_posture') if isinstance(detail.get('ingestion_posture'), str) else None,
        scenarios=scenarios,
        policy_tiers=policy_tiers,
        documents=_documents(detail.get('documents')),
    )
