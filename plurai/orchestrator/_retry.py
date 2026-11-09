"""Risk-scoped HTTP retry policy for :mod:`plurai.orchestrator.client` (tenacity).

Retry policy is split by failure class because a retry can DOUBLE-APPLY a
non-idempotent POST (create_thread / finalize / fail) when the request DID
reach the server but the response/ack was lost:

- Connection-phase failures (:exc:`httpx.ConnectError`,
  :exc:`httpx.ConnectTimeout`) are retried for **all** requests — the
  request never reached the server, so a retry cannot double-apply.
- Read / write / pool timeouts (other :exc:`httpx.TimeoutException`), the other
  transport-level network failures (:exc:`httpx.NetworkError` — a socket that
  dies mid-response is a ``ReadError``, NOT a timeout, so it needs its own entry) and
  HTTP 502 / 503 / 504 are retried **only for idempotent** requests
  (``idempotent=True``): our GET endpoints, and ``/user-turn``, which the server
  makes idempotent behind a ``turn_idx`` key (a retried turn replays its stored
  reply instead of re-driving the dialog). For the remaining
  non-idempotent POSTs the server may have already applied the effect, so we
  surface the failure instead of risking a duplicate — save the ONE narrow
  ``retry_unforwarded`` widening below (which a turn_idx-less ``/user-turn`` still
  rides).
- HTTP 429 (a rate limiter refusing the request) is retried for idempotent requests,
  and for non-idempotent ones only under ``retry_unforwarded``: a rate-limiter refusal is
  answered before the request is forwarded. Its wait honours ``Retry-After`` (capped), and it
  has a budget of its own (:attr:`RetryConfig.throttle_retries`), so a short budget for 5xx
  does not give up on a throttle that a few seconds' wait would clear. The arithmetic is the
  one every SDK HTTP stack shares (:mod:`plurai._http_retry`).
  It is deliberately NOT in :data:`TRANSIENT_STATUS_CODES`, which also drives the re-drive
  classifier and the outage tag.

4xx other than 429 (client/auth errors), and non-gateway 5xx, are never retried. Back-off is
exponential with ADDITIVE jitter — ``min(cap, base * 2**attempt + uniform(0, base))``
— :func:`plurai._http_retry.backoff_seconds`, which every SDK HTTP stack shares.

A retryable 5xx means the SERVER needs a moment, so every wait has a real floor
(``base * 2**attempt``) and grows from one attempt to the next.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
)

from plurai._http_retry import (
    BACKOFF_CAP_SECONDS,
    RATE_LIMIT_RETRIES,
    RATE_LIMITED_STATUS,
    TRANSIENT_RETRIES,
    TRANSIENT_STATUS_CODES,
    backoff_seconds,
    is_name_not_found,
    parse_retry_after,
    throttled_wait,
)
from plurai.capacity import UnavailableBlock
from plurai.credit import CREDIT_REFUSED_STATUS, CreditBlock, credit_block_from_body, credit_block_or_unknown
from plurai.customer_safe import response_json
from plurai.network_refusal import NetworkRefusal, note, of_response, sentence
from plurai.orchestrator.models import FinalizeJobStatusResponse, RescoreJobStatusResponse

if TYPE_CHECKING:
    from plurai.settings import NetworkSettings

logger = logging.getLogger(__name__)

#: Transport failures that mean "the exchange did not complete", as opposed to "the server
#: answered". Retryable for an idempotent request, and the set the storage client and the ingest
#: command catch — named once so they cannot drift into disagreeing about what "transient" means.
#:
#: ``NetworkError`` is what makes a bare ``ReadError('')`` — a socket that died mid-response —
#: retryable; it is not a ``TimeoutException``, so it needs its own entry.
#: ``RemoteProtocolError`` is the server hanging up without a complete response. Deliberately
#: NOT ``TransportError`` wholesale: ``UnsupportedProtocol`` / ``LocalProtocolError`` are
#: programming errors that no amount of back-off will fix.
TRANSIENT_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    # A customer behind a corporate proxy sees proxy failures for the same reasons everyone
    # else sees network ones, and they resolve the same way. The exclusion rationale above is
    # about PROGRAMMING errors, which this is not.
    httpx.ProxyError,
)


def is_transient_for_idempotent(exc: BaseException) -> bool:
    """Whether *exc* means "the exchange did not complete" for a request that is safe to repeat.

    THE predicate — the per-request retry classifier below and the outage tag (:func:`outage_block_of`) both
    ask this exact question, and a second copy is how they come to disagree about what "transient" means.
    Covers the transport failures AND the gateway statuses: after ``request_with_retry`` exhausts its
    ~61-66s window a 502/503/504 surfaces as an ``HTTPStatusError``.

    NOT a general "should I retry" — the caller still owns idempotency. A non-idempotent POST
    must not consult this.
    """
    if isinstance(exc, TRANSIENT_TRANSPORT_ERRORS):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in TRANSIENT_STATUS_CODES


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Bounded retry parameters for :func:`request_with_retry`.

    ``max_retries`` is the number of **additional** attempts after the first;
    total attempts = ``max_retries + 1``. ``backoff_base`` is the FLOOR of the first
    wait and the exponential's multiplier; ``backoff_cap`` bounds it:
    ``min(cap, base * 2**attempt + uniform(0, base))``.

    The env-driven source of these values is :class:`~plurai.settings.NetworkSettings`
    (``NETWORK__RETRIES`` / ``NETWORK__BACKOFF_BASE`` / ``NETWORK__BACKOFF_CAP``),
    mapped on the production path via :meth:`from_settings`. The dataclass
    defaults below MIRROR the NetworkSettings field defaults for standalone
    construction (settings cannot be imported here at runtime — that would
    cycle through the observability sinks); a test fences the two against drift.
    """

    max_retries: int = TRANSIENT_RETRIES
    # These retries exist for a server that just failed a heavy request (``/finalize``
    # runs all of the backend's judges), so the first wait is a full second.
    #
    # 6 retries: the waits are ``min(base * 2**(n-1) + U(0, base), cap)``, so at base=1 and cap=30 they are
    # 1, 2, 4, 8, 16 and 30 (+ up to 1s jitter on the first five): a sleep window of ~61-66s, which outlasts the
    # window in which a backend upgrade rejects every request. The count and the cap size the window together.
    # Mirrors NetworkSettings' defaults (tests fence the two against drift).
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    #: Log a transient failure the loop rides out, and a transient budget spent, at DEBUG: this package never configures
    #: logging, so a WARNING reaches the terminal, and the run says nothing of an outage until it ends. A command that
    #: shows its retries sets this off. A throttle always warns.
    quiet_transient: bool = True

    #: The retries a 429 gets: the SDK-wide budget, whatever the transient window is. A count the caller chose
    #: (``--retries N``) raises it to at least that many (:meth:`from_settings`).
    throttle_budget: int = RATE_LIMIT_RETRIES

    @property
    def throttle_retries(self) -> int:
        """Retries a 429 gets: none when retries are off, else :attr:`throttle_budget`.

        A throttle is waited out on the server's schedule, not the caller's, so a config tuned for fast
        5xx retries (few attempts, a short cap) still gives a rate limiter the room it asked for.
        """
        return 0 if self.max_retries == 0 else self.throttle_budget

    @property
    def throttle_cap(self) -> float:
        """The longest a single 429 wait may be: the larger of this config's cap and the SDK-wide one."""
        return max(self.backoff_cap, BACKOFF_CAP_SECONDS)

    @classmethod
    def from_settings(cls, network: NetworkSettings) -> RetryConfig:
        """Map the ``NETWORK__*`` env-driven retry knobs onto a RetryConfig.

        ``retries`` left unset keeps the default transient window and the SDK-wide throttle budget; a count the caller chose
        is the transient budget, and the throttle gets at least that many.
        """
        if network.retries is None:
            return cls(backoff_base=network.backoff_base, backoff_cap=network.backoff_cap)
        return cls(
            max_retries=network.retries,
            backoff_base=network.backoff_base,
            backoff_cap=network.backoff_cap,
            throttle_budget=max(network.retries, RATE_LIMIT_RETRIES),
        )


#: Read, write and pool timeouts: slow, not down. The backend took the request and did not answer in time (a slow turn, or a
#: request that may already have been applied), or no pooled connection came free. None is an outage, whatever the retry
#: rule does with it; the finalize poll resumes by job id on them.
NON_OUTAGE_TIMEOUTS: tuple[type[BaseException], ...] = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)


def outage_block_of(exc: BaseException) -> UnavailableBlock | None:
    """The unavailable block a failed request shows: the connection failed (refused, reset, connect timeout) or the gateway
    answered 502, 503 or 504, and the answer is Plurai's own.

    A throttle is no outage, and neither is an answer from a proxy or firewall (:class:`NetworkRefused`), which keeps its own
    operator text, nor a timeout after the connection was made, nor a host name that does not resolve.
    """
    if isinstance(exc, (NetworkRefused, *NON_OUTAGE_TIMEOUTS)) or is_name_not_found(exc) or not is_transient_for_idempotent(exc):
        return None
    return UnavailableBlock('status' if isinstance(exc, httpx.HTTPStatusError) else 'transport')


def tag_outage(exc: BaseException) -> None:
    """Put the unavailable block of *exc* on it (:func:`~plurai.capacity.unavailable_block_of` reads it), when it shows an
    outage. Called on a request's final failure, and only when the failure is one the policy retries: a spent budget is an
    outage that outlasted what the client may do about it."""
    if (block := outage_block_of(exc)) is not None:
        cast('Any', exc).unavailable = block


def _is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == RATE_LIMITED_STATUS


def retry_after_seconds(response: httpx.Response, *, now: float | None = None) -> float | None:
    """The wait a response's ``Retry-After`` header asks for, in seconds, or ``None`` when it asks for none."""
    return parse_retry_after(response.headers.get('Retry-After'), now=now)


def _is_retryable(exc: BaseException, *, idempotent: bool, retry_unforwarded: bool) -> bool:
    """Classify an attempt failure under the risk-scoped policy above.

    Transient statuses (502/503/504) retry only for IDEMPOTENT requests: for a
    non-idempotent POST the request may have reached the server and been
    applied, so a status-error retry risks double-applying. No server contract
    narrows that risk for /user-turn: the app does not promise to emit 503 only
    after rolling back the turn's effects, so the general rule is the whole
    policy.

    For an idempotent request the retryable transport class is
    :data:`TRANSIENT_TRANSPORT_ERRORS` — timeouts AND the other incomplete-exchange failures.
    Retrying only ``TimeoutException`` would classify a bare ``ReadError('')`` on the
    finalize-job poll GET — an idempotent read of a job the SERVER had already finished — as
    non-retryable and kill the scenario.

    ``retry_unforwarded`` is the ONE narrow non-idempotent widening, justified by
    provenance rather than a rollback contract: a 503 on ``/threads/*`` comes only
    from the gateway refusing to forward, because the app itself has no 503 exit on
    those routes (the backend emits none there). A request the gateway never forwarded was provably NOT applied, so retrying it
    cannot double-apply. A 429 is the same kind of answer — a rate limiter refusing
    before it forwards — so it rides the same widening. 502/504/timeouts stay
    non-retryable for non-idempotent requests — those can occur mid-flight, after
    the request reached the app.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    if not idempotent:
        return retry_unforwarded and isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (503, RATE_LIMITED_STATUS)
    return is_transient_for_idempotent(exc) or _is_rate_limited(exc)


class ServerHTTPError(httpx.HTTPStatusError):
    """A non-2xx answer from the backend, typed where it is received (``kind='backend_http'``).

    Never the agent's. ``status`` is the answer and ``transient`` (5xx / 429) whether one more try could
    change it: a transient status is the backend's; any other is the backend REFUSING the request (the
    request, the run's configuration, or the backend itself), and ``error.json``'s ``response_body`` says
    which. An :exc:`httpx.HTTPStatusError`, so every existing handler still catches it.
    """

    kind: ClassVar[str] = 'backend_http'

    @property
    def status(self) -> int:
        return self.response.status_code

    @property
    def transient(self) -> bool:
        return self.status >= 500 or self.status == 429


class NetworkRefused(ServerHTTPError):  # noqa: N818  # a refusal a network control made, named like CreditRefused
    """A :class:`ServerHTTPError` whose answer did not come from Plurai (:func:`plurai.network_refusal.of_response`).

    Raised exactly where a :class:`ServerHTTPError` would be, with the same ``kind``, ``status`` and ``transient``; only the
    message differs (:func:`plurai.network_refusal.sentence`), and ``refusal`` carries the typed record.
    """

    def __init__(self, refusal: NetworkRefusal, *, request: httpx.Request, response: httpx.Response) -> None:
        super().__init__(sentence(refusal), request=request, response=response)
        self.refusal = refusal


class CreditRefused(Exception):  # noqa: N818  # a refusal the backend made, named like its status, not an error of ours
    """The backend refused the work for credit: ``credit`` names the kind and, for the free tier, the reset time.

    Raised for a 402 whose ``code`` is a credit kind, and for a finalize or rescore job that ended on a credit refusal
    (:func:`job_refused_for_credit`, :meth:`for_job`). Never transient: retrying, re-driving or resuming a poll cannot change what the
    organization's credit standing allows, and each attempt may spend credit the run does not have.

    It is not an :exc:`httpx.HTTPStatusError`: a job answer has no HTTP exchange to carry. ``response`` (and with it ``request``,
    ``status`` and ``request_path``) is set only when a real 402 came back, and is ``None`` for a job.
    """

    kind: ClassVar[str] = 'backend_credit'
    transient: ClassVar[bool] = False

    def __init__(self, message: str, *, credit: CreditBlock, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.credit = credit
        self.response = response

    @property
    def request(self) -> httpx.Request | None:
        """The request the 402 answered, or ``None`` for a job."""
        return self.response.request if self.response is not None else None

    @property
    def status(self) -> int | None:
        """The HTTP status of the refusal, or ``None`` for a job."""
        return self.response.status_code if self.response is not None else None

    @property
    def request_path(self) -> str | None:
        """The path of the refused request (``/threads``, ``/threads/{id}/user-turn``), or ``None`` for a job."""
        request = self.request
        return request.url.path if request is not None else None

    @classmethod
    def for_job(
        cls, job: FinalizeJobStatusResponse | RescoreJobStatusResponse, *, thread_id: str, job_id: str, route: Literal['finalize', 'rescore'] = 'finalize'
    ) -> CreditRefused:
        """The refusal of a finalize or rescore job that ended on a credit refusal.

        The kind is the ``credit`` block of the job's ``error``, else its ``detail`` when that is a known kind. A job that
        says credit without a readable kind is still credit: the sentinel kind.
        """
        error = job.error
        if error is not None and error.credit is not None:
            credit = credit_block_or_unknown(error.credit.model_dump())
        else:
            credit = credit_block_or_unknown({'code': error.detail} if error is not None else None)
        return cls(f'Async {route} job {job_id} on thread {thread_id} was refused for credit ({credit.code})', credit=credit)


def job_refused_for_credit(job: FinalizeJobStatusResponse | RescoreJobStatusResponse) -> bool:
    """Whether a failed finalize or rescore job ended on a credit refusal: its ``error`` says ``reason: credit``."""
    return job.error is not None and job.error.reason == 'credit'


def _raise_for_status(response: httpx.Response, *, endpoint: str = '') -> None:
    """:meth:`httpx.Response.raise_for_status`, raising :class:`ServerHTTPError`, :class:`NetworkRefused` or :class:`CreditRefused`.

    Raised outside the handler, so with no chained context.

    A 402 whose body names a credit kind raises :class:`CreditRefused`; a 402 with any other body stays a plain
    :class:`ServerHTTPError`. An answer that did not come from Plurai raises :class:`NetworkRefused` and is recorded
    (:func:`plurai.network_refusal.note`, which never raises) once the error is built. ``endpoint`` is the call site's
    route template, the record's ``request_kind``.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        credit = credit_block_from_body(response_json(exc.response)) if exc.response.status_code == CREDIT_REFUSED_STATUS else None
        error: ServerHTTPError | CreditRefused
        if credit is not None:
            error = CreditRefused(str(exc), credit=credit, response=exc.response)
        elif (refusal := of_response(exc.response, request_kind=endpoint or 'request')) is not None:
            error = NetworkRefused(refusal, request=exc.request, response=exc.response)
        else:
            error = ServerHTTPError(str(exc), request=exc.request, response=exc.response)
    else:
        return
    if isinstance(error, NetworkRefused):
        note(error.refusal)
    raise error


async def request_with_retry(
    call: Callable[[], Awaitable[httpx.Response]],
    *,
    config: RetryConfig,
    idempotent: bool,
    endpoint: str = '',
    retry_unforwarded: bool = False,
) -> httpx.Response:
    """Execute *call*, raise on non-2xx, and retry per the risk-scoped policy.

    Returns the successful response. On a non-retryable failure the exception
    (:exc:`httpx.HTTPStatusError`, :class:`CreditRefused` or a transport error) propagates immediately;
    on exhaustion the LAST failure is re-raised as-is.
    """
    max_attempts = config.max_retries + 1
    label = endpoint or 'request'
    method = 'GET' if idempotent else 'POST'
    # Failures so far, by budget: a throttle and every other retryable failure are counted apart.
    failed = {'throttled': 0, 'other': 0}

    def _throttled(exc: BaseException | None) -> bool:
        return exc is not None and _is_rate_limited(exc)

    def _log_before_sleep(retry_state: RetryCallState) -> None:
        reason = retry_state.outcome.exception() if retry_state.outcome is not None else None
        delay = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
        if _throttled(reason):
            logger.warning('throttled (HTTP 429) for %s %s; retry %d/%d in %.2fs', method, label, failed['throttled'], config.throttle_retries, delay)
            return
        logger.log(
            logging.DEBUG if config.quiet_transient else logging.WARNING,
            'attempt %d/%d for %s %s failed (%r); retrying in %.2fs',
            failed['other'],
            max_attempts,
            method,
            label,
            reason,
            delay,
        )

    def _log_exhausted_and_reraise(retry_state: RetryCallState) -> httpx.Response:
        if retry_state.outcome is None:  # pragma: no cover - tenacity always records the outcome
            raise RuntimeError('retry loop exhausted without recording a failure')
        spent = retry_state.outcome.exception()
        logger.log(
            logging.DEBUG if spent is not None and outage_block_of(spent) is not None and config.quiet_transient else logging.ERROR,
            'all %d attempt(s) for %s %s failed; surfacing the last failure',
            retry_state.attempt_number,
            method,
            label,
        )
        return retry_state.outcome.result()  # re-raises the captured exception

    async def _attempt() -> httpx.Response:
        try:
            response = await call()
            _raise_for_status(response, endpoint=label)
        except Exception as exc:
            failed['throttled' if _throttled(exc) else 'other'] += 1
            raise
        return response

    def _stop(retry_state: RetryCallState) -> bool:
        if _throttled(retry_state.outcome.exception() if retry_state.outcome is not None else None):
            return failed['throttled'] > config.throttle_retries
        return failed['other'] > config.max_retries

    def _wait(retry_state: RetryCallState) -> float:
        """A 429's wait is the back-off or what ``Retry-After`` asks, whichever is longer; any other failure backs off. Both capped."""
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        if exc is not None and _throttled(exc):
            backoff = backoff_seconds(failed['throttled'], base=config.backoff_base, cap=config.throttle_cap)
            return throttled_wait(backoff, retry_after_seconds(exc.response), cap=config.throttle_cap)  # type: ignore[attr-defined]
        return backoff_seconds(failed['other'], base=config.backoff_base, cap=config.backoff_cap)

    retrying = AsyncRetrying(
        retry=retry_if_exception(lambda exc: _is_retryable(exc, idempotent=idempotent, retry_unforwarded=retry_unforwarded)),
        stop=_stop,
        wait=_wait,
        before_sleep=_log_before_sleep,
        retry_error_callback=_log_exhausted_and_reraise,
    )
    try:
        return await retrying(_attempt)
    except Exception as exc:
        # Only a budget that was spent is an outage: the attempts made are ``max_retries + 1`` (so ``--retries 0`` makes one failure
        # the whole budget). A failure the policy does not retry (a create that may already have been forwarded) raised before
        # the budget was spent, waited for nothing, and stays a failed scenario.
        if failed['other'] >= max_attempts and _is_retryable(exc, idempotent=idempotent, retry_unforwarded=retry_unforwarded):
            tag_outage(exc)
        raise
