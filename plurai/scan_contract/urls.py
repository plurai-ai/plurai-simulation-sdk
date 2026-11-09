"""Derive the licensed-URL set from the scanned corpus, keeping TEMPLATES as patterns.

A documented URL is not always a literal. Product documentation routinely writes the customer's
own address as a template — ``https://<your-org>.example.com`` — and the documented, correct
behaviour for an agent is to reproduce that template OR to instantiate it for the user it is
talking to. An allowlist derived by copying literals licenses only the placeholder spelling, so
the instantiated host — the one a real user ever sees — is graded as a fabricated URL. That is a
FALSE REJECTION built at scan time and paid for in a failed run.

**This module owns DERIVATION, not matching.** Extraction and placeholder matching already exist
and are already what generated policies run:
the backend's URL extractor (markdown-aware; it preserves a ``<placeholder>`` that
follows ``/`` or ``.`` — see its ``_URL_TEMPLATE_SEGMENT``, which truncates other placements) and
the backend's URL matcher (a placeholder compiles to a bounded run that never
crosses ``/``, ``?`` or ``#``; the comparison is case-insensitive throughout). Re-implementing
either here would put two allowlist engines in one product with subtly different semantics, and
the eval that ships would be graded by the other one. So the scan-time derivation emits templates
in exactly the form ``url_matches`` consumes, and :meth:`LicensedUrl.licenses` decides membership
through :func:`plurai.scan_contract.closed_classes.url_licensed` — THE house rule, the same
callable the judge's closed-class detection and the sandbox helper run. Membership is not a second
spelling of that rule, so the scheme and ``www.`` variants every other consumer licenses are
licensed here too. The cosmetic widenings that rule carries are stated in the decision text of each
entry kind, published once under the allowlist's ``decisions``.

The walk covers every corpus unit's TEXT and its flattened structured FIELDS, with provenance for
each URL, and a stated compilation decision per entry kind — because an allowlist that silently
widened is worse than none.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from plurai.scan_contract._dialog_urls import extract_urls, is_url_template
from plurai.scan_contract.closed_classes import url_licensed
from plurai.scan_contract.corpus import CorpusUnit


_WIDENING = (
    'The COSMETIC widenings, which are the house URL rule (``closed_classes.url_licensed``) and not this '
    "module's own: the scheme, a leading ``www.``, a trailing slash on the path or the route, a plain "
    '``#anchor``, and the ``!`` of a hashbang route. A documented page also licenses its own parameterised '
    'reads (``?lang=en``) unless the documented form carries a query of its own, in which case the query is '
    'part of the identity'
)
LITERAL_DECISION = f'literal URL — licensed exactly as written (case-insensitively). {_WIDENING}'
TEMPLATE_DECISION = (
    'TEMPLATE — licensed as a PATTERN via url_matches, not as the literal placeholder spelling. Each '
    'placeholder matches a non-empty run that never crosses "/", "?" or "#": in the host that is one label '
    'sequence, in the path exactly one segment. The verbatim template stays licensed too — quoting the '
    f'documentation is not fabrication. {_WIDENING}'
)


@dataclass(frozen=True)
class LicensedUrl:
    """One licensed URL, and where in the corpus it was licensed from."""

    #: The URL exactly as the corpus writes it — placeholders intact. This IS the pattern:
    #: ``url_matches(candidate, template)`` is the membership test, here and in a generated eval.
    template: str
    #: Unit ids (and ``unit#field.path`` addresses) the URL was found in.
    provenance: tuple[str, ...] = ()

    @property
    def is_template(self) -> bool:
        """Decided by the SAME regex ``url_matches`` compiles against, never by a local copy.

        A ``'<' in template`` test reports a malformed token like ``https://x.com/a{`` as a
        template and publishes the pattern ``decision`` for it, while ``url_matches`` compiles it
        to plain equality — the two-engines-disagree failure this module exists to avoid. The
        corpus extractor constrains its own tokens, but ``extra_licensed`` is caller-supplied.
        """
        return is_url_template(self.template)

    @property
    def decision(self) -> str:
        """What licensing this entry means. Published once per kind under the allowlist's ``decisions``, so a widened allowlist is never silent."""
        return TEMPLATE_DECISION if self.is_template else LITERAL_DECISION

    def licenses(self, url: str) -> bool:
        """Whether this entry licenses ``url`` — through :func:`url_licensed`, the ONE house rule.

        Delegating keeps one spelling of the URL rule across consumers:
        ``LicensedUrl('https://docs.example.com/x').licenses('http://www.docs.example.com/x')`` is
        True here exactly where the rule every other consumer runs says True. The empty-path root
        equivalence is covered too: ``licence_form`` strips a trailing slash from the path, so the
        two root spellings are one address there.
        """
        return url_licensed(url, [self.template])

    def as_dict(self) -> dict[str, Any]:
        return {'template': self.template, 'is_template': self.is_template, 'provenance': list(self.provenance)}


@dataclass(frozen=True)
class UrlAllowlist:
    """The derived licensed-URL set, ordered by the case-folded dedup key so ordering is stable across spellings."""

    entries: tuple[LicensedUrl, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def templates(self) -> tuple[LicensedUrl, ...]:
        return tuple(e for e in self.entries if e.is_template)

    @property
    def literals(self) -> tuple[LicensedUrl, ...]:
        return tuple(e for e in self.entries if not e.is_template)

    def licensing(self, url: str) -> LicensedUrl | None:
        """The entry licensing ``url``, or ``None``.

        The allowlist is PUBLISHED (as templates, in the artifact) for a generated eval to match
        against via ``url_matches``. This is the membership test the derivation exists to support,
        and it shows that the published templates license what the decision strings claim.

        Literals are tried first so an exact hit is attributed to the URL it actually is rather
        than to a template that also happens to cover it — which matters only for the reported
        provenance, never for the verdict.
        """
        for entry in (*self.literals, *self.templates):
            if entry.licenses(url):
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        """The published form.

        An entry's decision is one of two fixed texts selected by ``is_template``, so they ride once under ``decisions``, keyed by kind.
        Repeating a long fixed text on every entry would grow a catalog's artifact toward the backend's document cap for no information."""
        return {
            'count': len(self.entries),
            'template_count': len(self.templates),
            'decisions': {'literal': LITERAL_DECISION, 'template': TEMPLATE_DECISION},
            'urls': [e.as_dict() for e in self.entries],
        }


def _is_bare_root(url: str) -> bool:
    """Two slashes — ``scheme://host`` with no path (``https://a.com``).

    Read only by :func:`_dedup_key`, to decide whether a trailing slash is cosmetic before two
    spellings of one address become two allowlist entries. A query- or fragment-only URL
    (``https://a.com?x=1``) also has two slashes and is treated as a root here; harmless, and
    cheaper than parsing, because the dedup only ever DROPS a slash the key does not need.
    """
    return url.count('/') == 2


def _dedup_key(url: str) -> str:
    """Case-folded, with an empty path's trailing ``/`` dropped — the two spellings name one
    address, so the allowlist must not carry both as separate entries."""
    text = url.strip()
    return (text[:-1] if text.endswith('/') and _is_bare_root(text[:-1]) else text).lower()


def derive_url_allowlist(units: Iterable[CorpusUnit], *, extra_licensed: Sequence[tuple[str, str]]) -> UrlAllowlist:
    """Every URL the scanned corpus licenses, with provenance.

    Both the unit TEXT and its flattened structured ``fields`` are read: a catalog's ``link_url``
    column is a licensed URL even when it never appears in prose, and a text-only derivation
    would miss exactly the entries a catalog exists to declare. ``extra_licensed`` is the
    ``(url, provenance)`` pairs licensed outside the corpus — the URLs the agent's own rules quote.
    Required, so no caller can leave that channel unwired by omission.
    """
    first_spelling: dict[str, str] = {}
    provenance: dict[str, list[str]] = {}

    def _record(url: str, source: str) -> None:
        key = _dedup_key(url)
        first_spelling.setdefault(key, url)
        provenance.setdefault(key, []).append(source)

    for unit in units:
        for url in extract_urls(unit.text):
            _record(url, unit.unit_id)
        for field_path, value in sorted(unit.fields.items()):
            for url in extract_urls(value):
                _record(url, f'{unit.unit_id}#{field_path}')
    for url, origin in extra_licensed:
        _record(url, origin)

    return UrlAllowlist(
        entries=tuple(LicensedUrl(template=first_spelling[key], provenance=tuple(dict.fromkeys(sources))) for key, sources in sorted(provenance.items()))
    )
