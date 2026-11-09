"""URL extraction + placeholder matching, a copy of the backend's URL helpers.

VENDORED — do not edit. The symbols below the marker are the backend's own bytes, and a parity check compares them with
the backend's source.

Only the symbols the scan contract's URL derivation needs are here. The upstream module also holds the transcript
projections, which import ``langchain_core`` at module level — a dependency this package must not acquire, since the
checker runs on a customer machine with pyyaml and nothing else.

``BaseMessage`` is therefore a local stand-in rather than the real class. It exists so ``message_texts`` — copied
verbatim below — keeps compiling: the scan contract only ever passes it ``str``, and the ``isinstance`` branch it guards
is unreachable from here. If a caller does hand this a langchain message, the object fails that check and contributes no
text; it never raises, and it never silently becomes a different kind of match.

``sys`` and ``unicodedata`` are here because the URL matcher's combining-mark class is DERIVED from Unicode's own
category data rather than hand-listed (a hand-listed one covers eight scripts and truncates every slug in the rest).
Both are stdlib, so the package stays installable with pyyaml and nothing else.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


class BaseMessage:  # noqa: B903 - a stand-in for an isinstance() branch, never constructed here
    """Not langchain's ``BaseMessage``. See the module docstring."""


MessageLike = BaseMessage | Mapping[str, Any]
MessageSource = str | MessageLike | Sequence[MessageLike] | None


# --- VENDORED VERBATIM from the backend's URL helpers ---------------------------------------


def _derive_combining_mark_ranges() -> str:
    """Every Unicode combining mark (categories Mn/Mc), as regex character-class ranges.

    DERIVED, not enumerated, and that is the whole point. A hand-listed class covers a fixed set
    of scripts, and a URL slug in any other script truncates at its first matra, as
    ``.../हिन्दी`` does when Devanagari is missing from the list. The failure is the one the block
    below describes: a STUB URL makes a membership allowlist admit every fabricated slug under it,
    a silent vacuous pass — repeated once per script nobody remembered to list.

    A table somebody has to remember to extend goes stale, so the class comes from
    ``unicodedata``. Computed ONCE at import. The full range is scanned, not only the
    BMP-plus-plane-1 span that holds every mark of every living script (the only marks above it
    are the plane-14 variation selectors): a bound chosen here is the same freeze this function
    exists to remove, one level down, and the one-off cost is not worth being wrong about a
    script.
    """
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    previous = 0
    for code in range(sys.maxunicode + 1):
        if unicodedata.category(chr(code)) in ('Mn', 'Mc'):
            if start is None:
                start = code
            previous = code
        elif start is not None:
            ranges.append((start, previous))
            start = None
    if start is not None:
        ranges.append((start, previous))
    # re.escape on each endpoint, not on the joined string: a class body is not escapable as a
    # unit, and a lone '-' or ']' endpoint would otherwise re-open the class it sits in. Every
    # mark is non-ASCII today, so this escapes nothing — it is here so a future Unicode release
    # cannot turn this into an injection.
    return ''.join(re.escape(chr(lo)) if lo == hi else f'{re.escape(chr(lo))}-{re.escape(chr(hi))}' for lo, hi in ranges)


# URL matcher for extract_urls. The alternation's placeholder branches admit a
# PLACEHOLDER segment mid-URL in either common spelling (`<host>`/`<article-id>`
# or `{org}`/`{id}`) so templated URLs the docs teach
# (`https://<your-org>.example.com`, `https://{org}.example.com`) survive
# extraction instead of dying at the delimiter — a returned token containing
# '<' or '{' is a TEMPLATE, not a live URL. Placeholder SPELLING is the doc
# author's choice, so both are first-class here.
#
# A placeholder is recognized by its POSITION: it must open right after a URL
# structural delimiter (`//`, `/` or `.`), which is where a parameterized host label
# or path segment can appear. HTML the agent legitimately writes never sits there —
# the `<br>` in `https://a.example/x<br>then` follows `x`, so it is markup, not a
# segment. Without this the branch matched through `<br>` and kept consuming into the
# following word, failing a compliant reply against its own allowlist. Content is
# restricted to identifier characters, so `</p>` and `</li>` are excluded outright.
#
# The plain-character branch is a POSITIVE class, not a negated one. A negated class such as
# `[^\s<>"'()\[\]{}]` treats every character it doesn't name as URL material —
# including all non-ASCII and the backslash — so a Hebrew reply glues the following
# word onto the URL ('https://docs.example.comאם') and a literal '\n\n'
# escape sequence (manufactured when list-shaped message content is repr-
# stringified into the transcript) is swallowed whole, false-failing exact-match
# membership checks on any multilingual agent. Only characters a URL may legally
# carry match; quotes/brackets/parens are excluded.
#
# AUTHORITY and PATH carry different classes, because the two multilingual hazards
# pull in opposite directions and POSITION is what tells them apart:
#
# - After a bare host, non-ASCII is usually the NEXT WORD, not the URL. RTL text
#   writes 'https://docs.example.comאם תרצו' with no space, and it writes
#   'https://docs.example.com.אם תרצו' with the sentence's own full stop in
#   between — so admitting non-ASCII freely re-creates the glue bug above, which
#   costs a FALSE RED (a compliant URL fails its own allowlist with a Hebrew word
#   welded on).
#   A non-Latin host is separated from that by what precedes it: it opens the
#   authority (right after '//' or a placeholder's '>'), or it continues a host
#   whose previous label was itself non-Latin ('пример.рф'). So a non-ASCII
#   authority character is admitted only when neither an ASCII alphanumeric nor an
#   ASCII-label boundary ('.' or '-' after an ASCII alphanumeric) immediately
#   precedes it. The deliberate cost: a non-Latin TLD on an ASCII host
#   ('example.орг') is not extracted. That direction is chosen on purpose — an
#   un-extracted URL is a finding not made, while glue fails a correct agent.
#   Requiring the whole authority to be ASCII is not an option: when the
#   non-Latin label is the FIRST one the match dies at 'https://' and the URL is
#   invisible to every check rather than unlicensed by them (for example
#   'https://<שם-המוסד>.portal.example.com' extracts as no URL
#   at all, and the reply's membership check reads clean).
# - Once a path/query/fragment has opened, non-ASCII is the URL — an IRI (RFC 3987)
#   slug: 'https://www.example.com/products/מוצר-לדוגמה'. Cutting it at the
#   first Hebrew letter yielded the STUB 'https://www.example.com/products/', which
#   is not merely lossy: a membership allowlist that legitimately carries the
#   collection URL then admits every FABRICATED slug under it (silent vacuous pass),
#   and a reasoning_expression shows the reviewer a URL the agent never wrote.
#   Every non-Latin locale hits this.
#
# The path class is `\w` (Unicode-aware in Python 3) plus the same sub-delimiters, so it
# admits letters and digits of any script while still excluding whitespace, the backslash,
# quotes/brackets/parens, bidi and zero-width controls (category Cf is not a word character)
# and non-ASCII punctuation such as an em dash — all of which terminate the token.
#
# `\w` alone is NOT script-general, which is easy to miss: it is `str.isalnum()`-based, and
# COMBINING MARKS (categories Mn/Mc) are not alphanumeric. Unpointed Hebrew happens to be all
# base letters, so the Hebrew case looked general while `.../हिन्दी` truncated at the first
# matra and `.../العربيّة` at the shadda. The class below re-admits EVERY combining mark, in
# every script, so a slug in any of them survives — see `_derive_combining_mark_ranges` for why
# it is derived from `unicodedata` rather than listed. A hand-written list covers a fixed set of
# scripts and is wrong about the rest.
_COMBINING_MARKS = _derive_combining_mark_ranges()


#: A placeholder's NAME is prose the doc author wrote, so it carries whatever script the
#: docs are in — the Hebrew '<שם-המוסד>' is the same construct as
#: '<your-org>'. Restricted to word characters plus '.' and '-' exactly as the ASCII
#: spelling was; '</p>' is excluded by the '/', and what keeps '<br>' out is the
#: POSITION rule below, not this class.
_URL_TEMPLATE_SEGMENT = rf'(?<=[/.])(?:<[\w{_COMBINING_MARKS}.-]+>|\{{[\w{_COMBINING_MARKS}.-]+\}})'


#: A non-ASCII authority character, admitted at exactly three positions — see the AUTHORITY
#: note above. An ALLOWLIST of positions, never a denylist of glue characters: the authority
#: class carries ',', ':', '@', '!' and the rest of the sub-delimiters, so "not after an ASCII
#: letter" leaves a door open at every one of them and RTL prose walks through it.
_URL_AUTHORITY_NON_ASCII = (
    r'(?:'
    r'(?<=[/>])'  # opening the authority, or just past a placeholder segment
    r'|(?<![\x00-\x7f])'  # continuing a non-Latin label
    r'|(?<=[^\x00-\x7f].)'  # a new label whose PREVIOUS label was non-Latin
    r')'
    rf'(?:[^\W\x00-\x7f]|[{_COMBINING_MARKS}])'
)


#: An RFC 3986 scheme, GENERIC rather than `https?`. The allowlist derived from a corpus is only
#: as wide as what this matcher sees, so every scheme it does not know is a URL the docs license
#: and the derivation drops — and an agent reproducing a documented `wss://` endpoint or a
#: `myapp://` deep link is then graded as having fabricated it. That is the false rejection the
#: derivation exists to remove, one scheme down. Nothing in prose produces `word://`, so widening
#: the HIERARCHICAL form costs no precision at all.
_URL_SCHEME = r'[A-Za-z][A-Za-z0-9+.\-]*'


_URL_RE = re.compile(
    # Hierarchical form (`scheme://authority/path`).
    rf'{_URL_SCHEME}://'
    # Authority: ends at the first '/', '?' or '#'.
    rf'(?:{_URL_TEMPLATE_SEGMENT}|[A-Za-z0-9\-._~:@!$&*+,;=%]|{_URL_AUTHORITY_NON_ASCII})+'
    # Path / query / fragment: same delimiters, but any script's characters.
    rf'(?:[/?#](?:{_URL_TEMPLATE_SEGMENT}|[\w{_COMBINING_MARKS}\-._~:/?#@!$&*+,;=%])*)?'
    # OPAQUE form (`mailto:`, `tel:`, `sms:`, `xmpp:`) — no '//', so what follows the colon is
    # not an authority and the branch above cannot see it. Gated on SHAPE, never on a scheme
    # table: the body must carry an '@' (mail-shaped) or open with '+' and a digit (dialable).
    # Prose writes 'Note:see' and 'TODO:fix' with neither, and 'Contact: help@example.com' has the
    # space the negative lookahead refuses — so a sentence can never be extracted as a URL. The
    # deliberate cost: an opaque scheme of neither shape (`bitcoin:1A2b`, `urn:isbn:…`) is still
    # missed. That is the same direction the AUTHORITY note above chooses — an un-extracted URL is
    # a finding not made, while a `Note:see` extracted as a URL LICENSES a token no corpus wrote.
    rf'|{_URL_SCHEME}:(?![/\s])(?=[^\s<>"\'()\[\]{{}}]*@|\+\d)'
    rf'(?:[\w{_COMBINING_MARKS}\-._~:@!$&*+,;=%?#/]|{_URL_TEMPLATE_SEGMENT})+'
)


# A PURE autolink (`<https://…>`, no inner angle brackets) is unwrapped before
# matching so its closing '>' can never be confused with a template segment.
# Scheme-generic for the same reason `_URL_RE` is: `<wss://…>` and `<mailto:…>` are the
# markdown spelling of exactly the links the derivation must not drop.
_AUTOLINK_RE = re.compile(rf'<({_URL_SCHEME}:[^<>\s]+)>')


#: A markdown link whose LABEL is its own HREF — ``[https://x](https://x)`` — collapsed to the href
#: alone by :func:`extract_urls` (see its docstring for the counting bug this closes). The label
#: must be scheme-prefixed and may then contain anything but whitespace and the bracket/paren
#: delimiters that would end it, so templated URLs (``<org>``) and query strings survive; either
#: occurrence may sit in CommonMark angle brackets, the label may be emphasis-wrapped, and the
#: destination may carry a title — the shapes models actually write. The backreference demands
#: an EXACT repeat of the URL itself, so a label that differs from the href at all is not this
#: shape. Runs BEFORE the autolink unwrap (whose inserted space would otherwise break the repeat).
#: A closing '>' is consumed only on a side that opened with '<' (conditional groups), so a
#: templated href that ENDS in '>' (``…/users/<id>``) keeps its own bracket.
_SELF_LABELLED_LINK_RE = re.compile(
    r'\[\s*(?P<wrap>[*_~]{0,3})(?P<lo><)?(?P<href>[A-Za-z][A-Za-z0-9+.\-]*:[^\s\[\]()]+?)(?(lo)>)(?P=wrap)\s*\]'
    r'\(\s*(?P<do><)?(?P=href)(?(do)>)\s*(?:"[^"]*"|\'[^\']*\')?\s*\)'
)


def message_texts(source: MessageSource) -> list[str]:
    """Text content of ``source`` — a raw string, a single message, or a message list
    in any transcript shape (langchain objects, serialized dicts, Anthropic
    content-block lists)."""
    if isinstance(source, str):
        return [source]
    messages: Sequence[MessageLike] = [source] if isinstance(source, BaseMessage | Mapping) else source or []
    texts: list[str] = []
    for msg in messages:
        content = msg.get('content') if isinstance(msg, Mapping) else getattr(msg, 'content', None)
        if isinstance(content, list):
            texts.append(' '.join(b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text'))
        elif isinstance(content, str):
            texts.append(content)
    return texts


def extract_urls(source: MessageSource) -> list[str]:
    """Every URL in ``source`` (a string OR a message list), in order.

    ANY scheme, not only http(s): a corpus documenting ``wss://``, ``mailto:``, ``tel:`` or a
    ``myapp://`` deep link licenses those URLs, and a matcher that cannot see them derives an
    allowlist that grades an agent as fabricating what its own documentation told it to write.
    The opaque (no ``//``) form is admitted on SHAPE — see the ``_URL_RE`` comment — so a prose
    colon is never extracted as a URL.

    URL allowlist/membership policies must use this instead of hand-rolled
    ``.split()`` tokenization: a markdown-embedded URL
    (``[![alt](https://img)](https://link)``) is ONE whitespace token starting
    with ``[![``, so ``token.startswith('http')`` never fires and the policy
    passes vacuously. The regex cuts at whitespace and markdown/HTML delimiters
    (so bare, ``[text](url)``, ``![img](url)``, and ``<url>`` forms all
    surface) and trailing sentence punctuation is stripped. Caveat: a URL
    whose PATH itself contains ``()[]{}`` or quotes (rare — e.g. Wikipedia
    disambiguation pages) is truncated at that character; compare such
    allowlist entries with ``startswith`` instead of equality. Pass
    ``agent_messages`` to scan only the agent's own words.

    Placeholder (templated) URLs are extracted too: a ``<...>`` segment inside a
    URL (``https://<your-org>.example.com/reset``) is kept as part of the token,
    so documentation that teaches a parameterized URL is visible to policies. A
    returned token containing ``'<'`` is a TEMPLATE, not a live URL — match live
    URLs against it with the ``url_matches`` helper, never string equality.

    ONE visible link is ONE URL. A markdown link whose label IS its own href
    (``[https://x](https://x)`` — the shape every model writes when asked to
    "give the link") renders as a single link, and a reader sees one URL. Without
    this collapse the regex returns it TWICE (label + href), so a per-reply cap
    (``len(...) <= 1``) counts a compliant single link as two and red-flags
    every reply that carries one. The collapse is exact-match only: ``[text](url)`` is
    already one URL, ``[![alt](img)](link)`` is genuinely two, and a label that
    differs from its href by so much as a trailing slash is left as two tokens
    (an allowlist compares each; a cap should count ``len(set(...))``)."""
    urls: list[str] = []
    for text in message_texts(source):
        # Collapse self-labelled links FIRST (see the docstring): the label is dropped and the
        # href kept. Spaces on BOTH sides: the scheme class is generic, so a href glued to the
        # word before it (`see[https://a](https://a)`) would otherwise extract as `seehttps://a`.
        unwrapped = _SELF_LABELLED_LINK_RE.sub(r' \g<href> ', text)
        # Then unwrap pure autolinks (`<https://…>`) so their closing '>' can't interact with
        # the template branch of _URL_RE. Spaced on both sides for the same reason.
        unwrapped = _AUTOLINK_RE.sub(r' \1 ', unwrapped)
        # Strip trailing sentence punctuation AND markdown emphasis markers (**url**,
        # *url*, ~~url~~) — emphasis-wrapped links are common agent output, and a
        # retained '**' flips a compliant allowlist check to violated. '*'/'~' are
        # legal URL chars but vanishingly rare as the FINAL character. The match
        # always starts with 'http', so stripping can never empty it.
        urls.extend(match.rstrip('.,;:!?*~_') for match in _URL_RE.findall(unwrapped))
    return urls


# Placeholder segment in a URL pattern: `<...>` or `{...}`. Kept module-level so
# url_matches' tokenizer and the template test used by callers stay in one place.
_URL_PLACEHOLDER_RE = re.compile(r'<[^<>]+>|\{[^{}]+\}')


def is_url_template(pattern: str) -> bool:
    """Whether ``pattern`` carries a placeholder — i.e. whether :func:`url_matches` compiles it
    as a PATTERN rather than degrading to case-insensitive equality.

    Public because callers deriving an allowlist need to label an entry as a template, and a
    local ``'<' in pattern`` test answers a different question: it calls ``https://example.com/a{`` a
    template while ``url_matches`` treats it as a literal. Two answers to "is this a pattern?" in
    one product is how an allowlist and the eval graded against it come apart.
    """
    return isinstance(pattern, str) and _URL_PLACEHOLDER_RE.search(pattern) is not None


def url_matches(url: str, pattern: str) -> bool:
    """Whether ``url`` matches ``pattern``, a URL whose ``<...>`` / ``{...}``
    segments are placeholders; everything else is literal.

    ONE uniform placeholder rule, wherever the placeholder sits: it matches a
    non-empty run of characters that never crosses ``/``, ``?`` or ``#``. That
    reads differently by position without needing to be defined per position:

    - in the HOST it is one label sequence (a dot is an ordinary character):
      ``url_matches('https://acme.eu.example.com/x', 'https://<org>.example.com/x')``
      → True, while ``https://evil.example/x`` → False.
    - in the PATH it is exactly one segment:
      ``url_matches('https://kb.example.com/articles/12345',
      'https://kb.example.com/articles/<article-id>')`` → True, but
      ``.../articles/12/34`` → False.
    - the comparison is case-insensitive THROUGHOUT — scheme, host and path
      alike: a policy checking what an agent wrote in prose should not fail a
      compliant link over the case a model typed.
    - a pattern with NO placeholder degrades to case-insensitive equality.
    - a non-string or malformed token never raises — it just fails to match, so a
      policy expression cannot error on a weird token.

    The canonical allowlist shape (see the generate-evals recipe)::

        all(u in ['<literal>', ...] or url_matches(u, 'https://<org>.example.com/reset')
            for u in extract_urls(agent_messages))
    """
    if not isinstance(url, str) or not isinstance(pattern, str):
        return False
    pattern = pattern.strip()
    pieces: list[str] = []
    last = 0
    for m in _URL_PLACEHOLDER_RE.finditer(pattern):
        pieces.append(re.escape(pattern[last : m.start()]))
        pieces.append(r'[^/?#]+')
        last = m.end()
    pieces.append(re.escape(pattern[last:]))
    return re.fullmatch(''.join(pieces), url.strip(), re.IGNORECASE) is not None
