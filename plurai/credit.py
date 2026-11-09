"""The credit refusal vocabulary: the kinds, the block that carries one, and how it is read off a body or an error.

A credit refusal is the backend declining work because the organization's credit standing does not allow it. The
backend owns the contract and sends data only: a kind (``code``) and, for the free tier, the UTC time the allowance
resets (``resets_at``, ``%Y-%m-%dT%H:%M:%SZ``). The block has the shape and field names of the ``capacity`` block, so
one reader serves both. No sentence for the user crosses the wire; the words belong to the caller.

Importing nothing from the rest of the package at module load, so every layer (the HTTP client, the runner, the
report) can share the one closed set of kinds; :func:`credit_refusal_sentence` imports its two helpers where it uses them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, get_args

if TYPE_CHECKING:
    import httpx

CreditKind = Literal['insufficient-credit', 'new-run-refused', 'free-tier-ceiling', 'credit-unavailable', 'credit-refused']

#: Every kind the backend sends. ``credit-refused`` is the backend's own answer for a refusal whose kind it could not
#: read, so a caller never has to invent one for "credit, kind unknown".
CREDIT_KINDS: Final[tuple[CreditKind, ...]] = get_args(CreditKind)

#: The status the REST routes and the gateway's proxy answer a credit refusal with, for every kind.
CREDIT_REFUSED_STATUS: Final = 402

#: The kind for a credit refusal that names none.
UNKNOWN_KIND: Final[CreditKind] = 'credit-refused'


class CreditBlock(NamedTuple):
    """One credit refusal: its kind, and when the free tier resets (``None`` for every other kind)."""

    code: CreditKind
    resets_at: str | None = None

    def to_record(self) -> dict[str, str]:
        """The block as JSON-ready fields; ``resets_at`` is absent, never ``null``, when there is none."""
        return {'code': self.code} if self.resets_at is None else {'code': self.code, 'resets_at': self.resets_at}


def credit_block_from_body(body: object) -> CreditBlock | None:
    """The block of a problem-details body, or ``None`` unless its ``code`` is exactly one of :data:`CREDIT_KINDS`.

    Strict on purpose: a code the backend did not send is not a credit refusal, and a caller that guesses one would
    stop a run for a fault no top-up fixes. A ``resets_at`` that is not a non-empty string is dropped; the gateway's
    own refusal spells it ``resetsAt``, which is read when ``resets_at`` is absent.
    """
    if not isinstance(body, Mapping):
        return None
    code: Any = body.get('code')
    if not isinstance(code, str) or code not in CREDIT_KINDS:
        return None
    return CreditBlock(code, resets_at_of(body))  # type: ignore[arg-type]


def resets_at_of(body: object) -> str | None:
    """The reset time of a problem-details body in wire form, or ``None``.

    ``resets_at`` is taken as sent. The gateway's own refusal spells the field ``resetsAt``, with milliseconds and any
    offset; it is read when ``resets_at`` is absent and re-emitted in the wire form (UTC, whole seconds).
    """
    if not isinstance(body, Mapping):
        return None
    resets_at = body.get('resets_at')
    if isinstance(resets_at, str) and resets_at.strip():
        return resets_at
    moment = _utc_moment(body.get('resetsAt'))
    return moment.strftime(_WIRE_TIME) if moment is not None else None


def credit_block_or_unknown(body: object) -> CreditBlock:
    """:func:`credit_block_from_body`, reading a body that names no known kind as :data:`UNKNOWN_KIND`.

    For a caller that already knows the refusal is a credit one (its cause or terminal reason says so) and only needs the kind.
    """
    return credit_block_from_body(body) or CreditBlock(UNKNOWN_KIND)


def credit_block_of(exc: BaseException) -> CreditBlock | None:
    """The credit block *exc* carries, or ``None``: one reader for the HTTP error and the finalize-job error alike."""
    block = getattr(exc, 'credit', None)
    return block if isinstance(block, CreditBlock) else None


#: What a sentence points the user at when the environment has no billing page of its own.
BILLING_FALLBACK: Final = 'the Plurai dashboard'

#: The contact hand-off: what the customer reads for a fault on Plurai's side that Plurai must look at, product's words
#: verbatim and one line.
CONTACT_TEXT: Final = (
    'We hit an issue that needs a closer look from the Plurai team. '
    "Reach out at support@plurai.ai or on Discord: https://discord.gg/bg34gh76uJ and we'll dig into it."
)

#: The one sentence per kind, as a template. The caller owns the words; the backend sends none. Placeholders:
#: ``{billing}`` the billing page address or :data:`BILLING_FALLBACK`; ``{until}`` `` until <time> UTC`` when the free
#: tier's reset time is known, else nothing; ``{after}`` ``after that`` or ``when it resets``, to match.
#: Only the kinds a payment fixes name the billing page (``insufficient-credit``, ``new-run-refused`` as its first
#: option, ``free-tier-ceiling`` as the upgrade); ``credit-refused`` asks the user to try again and nothing more: it
#: names no link, error id or contact. ``credit-unavailable`` is a fault on Plurai's side: it is :data:`CONTACT_TEXT`,
#: which does not mention credit at all. No figure, no status code and no mention of a quota appears.
CREDIT_SENTENCES: Final[Mapping[CreditKind, str]] = MappingProxyType(
    {
        'insufficient-credit': "Your organization has no credit left. Add credit at {billing}, or ask your organization's admin to.",
        'new-run-refused': (
            "Your organization's credit only allows a few runs at once, and work already running, which can include your "
            'own earlier steps, is using them. Add credit or raise the pay-as-you-go limit at {billing}, or wait for that '
            'work to finish and run fewer scenarios at a time.'
        ),
        'free-tier-ceiling': 'The free tier is used up{until}. Continue {after}, or upgrade at {billing}.',
        'credit-unavailable': CONTACT_TEXT,
        'credit-refused': 'Credit was refused. Please try again later.',
    }
)

#: The wire form of ``resets_at`` and the form a sentence shows it in.
_WIRE_TIME: Final = '%Y-%m-%dT%H:%M:%SZ'
_SHOWN_TIME: Final = '%Y-%m-%d %H:%M UTC'


def _utc_moment(resets_at: object) -> datetime | None:
    """*resets_at* as a UTC instant, or ``None`` when it is not a timestamp that names its offset.

    Any ISO 8601 form with an offset (``Z``, ``.000Z``, ``+00:00``, ``+02:00``) is read and converted. A time with no
    offset, text that is no timestamp and an instant that cannot be shown in UTC are ``None``.
    """
    text = resets_at.strip() if isinstance(resets_at, str) else ''
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text)
        return moment.astimezone(UTC) if moment.tzinfo is not None else None
    except (ValueError, OverflowError):
        # An offset that carries the instant outside the representable years cannot be shown in UTC.
        return None


def _shown_reset(resets_at: str | None) -> str | None:
    """*resets_at* as a UTC time for a sentence; the backend's own string never reaches the terminal."""
    moment = _utc_moment(resets_at)
    return moment.strftime(_SHOWN_TIME) if moment is not None else None


def credit_sentence(code: CreditKind, *, resets_at: str | None = None, billing_url: str = '') -> str:
    """The sentence for a credit refusal of kind *code*, from :data:`CREDIT_SENTENCES`.

    A kind the table does not hold reads as :data:`UNKNOWN_KIND`. *billing_url* is the environment's billing page (empty
    when it has none, which prints :data:`BILLING_FALLBACK`); it is only ever printed, never placed in a query string.
    *resets_at* matters to the free tier alone and is shown in UTC; a value that is not a timestamp with an offset reads
    as no reset time.
    """
    template = CREDIT_SENTENCES.get(code) or CREDIT_SENTENCES[UNKNOWN_KIND]
    reset = _shown_reset(resets_at)
    return template.format(
        billing=billing_url.strip() or BILLING_FALLBACK,
        until=f' until {reset}' if reset else '',
        after='after that' if reset else 'when it resets',
    )


def credit_refusal_sentence(response: httpx.Response) -> str | None:
    """The sentence for a 402 whose body names a credit kind, else ``None``.

    The billing page in it is the one of the backend that answered.
    """
    from plurai.customer_safe import response_json  # noqa: PLC0415 - keeps this module a leaf at import time
    from plurai.login._environments import billing_url_for  # noqa: PLC0415 - same

    if response.status_code != CREDIT_REFUSED_STATUS or (block := credit_block_from_body(response_json(response))) is None:
        return None
    return credit_sentence(block.code, resets_at=block.resets_at, billing_url=billing_url_for(str(response.url)))


__all__ = [
    'BILLING_FALLBACK',
    'CONTACT_TEXT',
    'CREDIT_KINDS',
    'CREDIT_REFUSED_STATUS',
    'CREDIT_SENTENCES',
    'UNKNOWN_KIND',
    'CreditBlock',
    'CreditKind',
    'credit_block_from_body',
    'credit_block_or_unknown',
    'resets_at_of',
    'credit_block_of',
    'credit_refusal_sentence',
    'credit_sentence',
]
