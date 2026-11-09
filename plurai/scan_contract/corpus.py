"""The scanned corpus as a TYPED, HASHABLE object — the input every scan-contract check reads.

Two things live here, and they are deliberately the same module because they are two views of
one fact:

* :class:`CorpusUnit` — one scanned source file, with its text and (for a structured file) its
  flattened scalar fields. Checks quote a unit id, so the id must be stable across scans: it is
  the source path relative to the scan root. A check that needs finer granularity addresses a
  FIELD (``assets.yaml#assets.2.link_url``) rather than a differently-cut unit.
* :class:`SourceInventory` — the per-file digests plus ONE ``content_hash`` over all of them.
  Without it the corpus scope is only a dated assertion, so a source edited after the scan is
  invisible to everything downstream. A hash stored beside ``corpus_id`` turns that assertion
  into a check.

Structured files (``.yaml`` / ``.yml`` / ``.json``) are flattened into ``fields`` — dotted path
-> scalar — rather than parsed against any schema. That is what keeps the module GENERIC: a
catalog's ``link_url`` becomes ``fields['assets.0.link_url']`` without this code knowing that
any catalog, or any customer, exists.
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

from plurai.scan_contract._safe_yaml import safe_yaml_load

#: Files admitted into a corpus scan by default. Text-shaped only: the checks read text, and a
#: binary admitted here would hash fine and then match nothing, which reads as coverage.
#:
#: DELIBERATELY NOT WIDENED, and this is the load-bearing half. A customer's help corpus is
#: routinely a Zendesk/Confluence/Notion export (``.html``), a catalog (``.csv``), or video
#: captions (``.srt``) — none of which are here, so one ``README.md`` beside many ``.html`` pages
#: loads ONE unit and every ``kind: absence`` fact is then VERIFIED over it, minting an
#: adversarial probe that grades "the agent must decline" against a corpus that documents the
#: thing many times. Adding those suffixes to this tuple would fix that scan and silently rewrite
#: every OTHER scan's content hash, drift baseline and absence verdict. So the miss is made LOUD
#: instead of guessed away: ``load_corpus`` counts what it skipped PER EXTENSION and
#: warns, and ``--suffixes`` admits them for the scan that wants them.
#:
#: This module owns no format table and no converter. The SDK, which does, passes its own
#: ``readers.SCAN_SUFFIXES`` with a :class:`SourceReader` that converts them; this default stays the
#: text-only scope of a scan with no reader plugged in.
DEFAULT_SOURCE_SUFFIXES: tuple[str, ...] = ('.md', '.markdown', '.txt', '.yaml', '.yml', '.json', '.rst')

#: How many per-extension skip warnings are emitted before the rest collapse into one line. An
#: agent repo used as a corpus root holds dozens of extensions; the warning has to be loud about
#: what was missed, not a page of them. Ordered by COUNT, so the 500 skipped ``.html`` pages are
#: never the line that got trimmed.
_MAX_SKIPPED_SUFFIX_WARNINGS = 8

#: Directory names never walked. Not a security boundary — a noise filter, so a hash does not
#: change every time a virtualenv is refreshed.
_SKIP_DIRS = frozenset({'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.mypy_cache', '.pytest_cache', '.ruff_cache'})

#: Structured suffixes get the ``fields`` flattening; everything else is text only.
_STRUCTURED_SUFFIXES = frozenset({'.yaml', '.yml', '.json'})

#: Share of a lossily-decoded source's characters that may be U+FFFD before the unit is called
#: UNDECODABLE rather than merely degraded. A single stray byte in an otherwise clean UTF-8 file
#: damages one character and leaves every check over it sound; a whole file in another encoding
#: (windows-1255 Hebrew is one example) turns every non-ASCII character into U+FFFD, and a
#: literal match over THAT reports "the corpus does not contain it" for every term written in
#: the encoding that was destroyed. The threshold is deliberately low: a source that lost a
#: tenth of its characters cannot support an absence claim either.
UNDECODABLE_REPLACEMENT_RATIO = 0.10

#: Leading bytes of the binary containers a corpus actually holds, and the name each is refused by.
#: The U+FFFD ratio does not catch them: a PDF's operator syntax is mostly ASCII and decodes as
#: near-valid UTF-8, so its ``obj … endobj`` text would be searched as facts with no warning.
_BINARY_MAGIC: tuple[tuple[bytes, str], ...] = ((b'%PDF-', 'pdf'), (b'PK\x03\x04', 'zip/office'), (b'\x1f\x8b', 'gzip'))


@functools.lru_cache(maxsize=1)
def _boundaryless_join_re() -> re.Pattern[str]:
    """Whitespace BETWEEN two characters of a script that writes without word spaces.

    Substituted away, because the whitespace it matches was put there by a line wrap or an indent
    and the script has no word boundary at it. Which scripts those are is ``facts``' knowledge —
    :data:`facts.BOUNDARYLESS_SCRIPTS`, with the class derived from its range table — not a second
    codepoint list kept here: two spellings of one script fact is how a range added to one is
    missed by the other. Built at first use rather than at import because ``facts`` imports
    :class:`CorpusUnit` from this module. A script the table does not name keeps a space at the
    wrap, which is correct for every spaced script.
    """
    from plurai.scan_contract.facts import boundaryless_character_class  # noqa: PLC0415 - import cycle: facts imports CorpusUnit from here

    cls = boundaryless_character_class()
    return re.compile(f'(?<=[{cls}]) +(?=[{cls}])')


def normalized(text: str) -> str:
    """Whitespace collapsed to single spaces — the ONE definition of the never-line-scoped rule.

    Every matcher in this package runs against this. Spelling it ``' '.join(x.split())`` at each
    call site is how one of them eventually does not.

    In a script that writes without inter-word spaces, collapsing a wrapped line to a single space
    would reintroduce the line scoping this function removes: a Chinese source wrapped mid-sentence
    would gain a space the writing system never has, so an existence fact about the joined term
    would come back CONTRADICTED (a manufactured red about text that is right there) and an
    absence fact about it would come back VERIFIED (a vacuous green). Whitespace between two
    characters of such a script is therefore dropped rather than collapsed; everywhere else the
    space is a real boundary and is kept.
    """
    return _boundaryless_join_re().sub('', ' '.join(text.split()))


#: Longest scalar admitted into ``fields``. A flattened long-form value is already in ``text``;
#: duplicating it into a field would double the memory of every structured corpus for nothing.
#: The cost is that such a field is NOT field-addressable, which changes the population a
#: field-scoped fact is decided over — so every drop is counted and warned about rather than
#: quietly shrinking the denominator.
_MAX_FIELD_CHARS = 2048


@dataclass(frozen=True)
class CorpusUnit:
    """One addressable unit of scanned corpus, with its text and its flattened scalar fields."""

    unit_id: str
    text: str
    #: Dotted-path -> scalar, for structured sources. Empty for prose. Checks may scope
    #: themselves to a field so an absence claim about a catalog column is not silently
    #: satisfied (or contradicted) by prose elsewhere in the same file.
    fields: Mapping[str, str] = field(default_factory=dict)
    #: The bytes did not decode as UTF-8 and the replacement-character density is past
    #: :data:`UNDECODABLE_REPLACEMENT_RATIO` — the text carried here is damaged, not the source.
    #: A matcher over it CANNOT establish that a term is absent: the term's own characters are
    #: exactly what the decode destroyed. Consumers must treat such a scope as unreadable rather
    #: than as searched-and-clean (see ``facts.verify_fact``).
    undecodable: bool = False

    @cached_property
    def _normalized_text(self) -> str:
        """The unit's whitespace-normalized text, computed once.

        ``verify_fact`` re-normalizes every unit for every fact, so this is O(facts x units)
        work over strings that never change — 77% of a 50-fact x 200-unit x 20 KB verification,
        measured. ``cached_property`` writes into ``__dict__`` directly, which is why it works on
        a frozen dataclass (``__slots__`` would break it).
        """
        return normalized(self.text)

    def scope_labels(self, field_name: str | None = None) -> list[str]:
        """Just the labels this unit contributes — no text, no normalization.

        The undecidable guards in ``verify_fact`` (behavioural claim, unknown kind, no terms)
        need only the COUNT of addressed scopes, and normalizing a corpus to populate an
        informational integer is the common case per this package's own docstring.
        """
        if field_name is None:
            return [self.unit_id]
        if any(ch in field_name for ch in '*?['):
            return [f'{self.unit_id}#{path}' for path in sorted(self.fields) if fnmatch.fnmatch(path, field_name)]
        return [f'{self.unit_id}#{field_name}'] if self.fields.get(field_name) else []

    def searchable(self, field_name: str | None = None) -> list[tuple[str, str]]:
        """``(label, text)`` pairs a matcher runs against — one per addressed scope.

        ``field_name`` is ``None`` for the whole unit, an exact dotted path for one field, or an
        fnmatch glob (``assets.*.link_url``) for a column across every record. A glob returns
        one pair PER FIELD rather than one concatenation of them all: concatenating is how a
        multi-term ``match: all`` gets satisfied by two different records, which is the
        cross-record version of the line-scoping bug and reads as a genuine hit.

        A scalar longer than :data:`_MAX_FIELD_CHARS` was never flattened and is therefore NOT
        addressable here — it is matchable only through the whole unit. ``load_corpus`` warns
        when that happens, because a field-scoped fact would otherwise be decided over a
        silently smaller population.

        Each text is whitespace-normalized, because matching raw makes every check LINE-SCOPED
        by accident: a claim spanning a wrapped markdown sentence stops matching the moment a
        source is reflowed, and the check then reports clean.
        """
        if field_name is None:
            return [(self.unit_id, self._normalized_text)]
        if any(ch in field_name for ch in '*?['):
            return [(f'{self.unit_id}#{path}', normalized(value)) for path, value in sorted(self.fields.items()) if fnmatch.fnmatch(path, field_name)]
        value = self.fields.get(field_name)
        return [(f'{self.unit_id}#{field_name}', normalized(value))] if value else []


@dataclass(frozen=True)
class SourceFile:
    """One scanned file's digest. ``path`` is relative to the scan root, POSIX-separated."""

    path: str
    sha256: str
    byte_count: int

    def as_dict(self) -> dict[str, Any]:
        return {'path': self.path, 'sha256': self.sha256, 'bytes': self.byte_count}


@dataclass(frozen=True)
class VisibleProjection:
    """What the agent's own prompt compiler actually PRESENTS of the scanned sources.

    The scan records the customer's bytes. What the agent SEES of them can differ by a transform
    the scan knows about, because the scan is what found the compiler.
    A source file can carry media URLs the compiler strips; a judge injected the raw file then
    writes "the ground truth specifies … the exact URL" — a URL the agent cannot know, graded as
    grounded.

    ``transform`` NAMES the projection (``media URLs stripped from every step``).
    It is required and it is prose on purpose: what a compiler does is not derivable from the
    bytes, and a projection nobody can name is a projection nobody can check.

    ``withheld`` is the typed DELTA — the values present in the raw source and absent from what
    the agent sees. It is the fact a judge needs: not "the source says X" but "the source says X
    AND THE AGENT CANNOT SEE IT", which is the difference between a fabrication and a fact the
    agent had no way to state.
    """

    transform: str = ''
    files: tuple[SourceFile, ...] = ()
    content_hash: str = ''
    withheld: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            'transform': self.transform,
            'content_hash': self.content_hash,
            'file_count': len(self.files),
            'files': [f.as_dict() for f in self.files],
            'withheld': list(self.withheld),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisibleProjection:
        files = tuple(
            SourceFile(path=str(f.get('path', '')), sha256=str(f.get('sha256', '')), byte_count=int(f.get('bytes', 0) or 0))
            for f in (payload.get('files') or [])
            if isinstance(f, Mapping)
        )
        withheld = payload.get('withheld')
        return cls(
            transform=str(payload.get('transform', '') or ''),
            files=files,
            content_hash=str(payload.get('content_hash', '') or ''),
            withheld=tuple(str(x) for x in withheld if str(x).strip()) if isinstance(withheld, (list, tuple)) else (),
        )


@dataclass(frozen=True)
class SourceInventory:
    """Every scanned file's digest plus one hash over the set.

    ``content_hash`` is computed over ``path\\0sha256`` lines in sorted path order, so it is
    stable across filesystems and walk order. Keying on PATH is what a hash over concatenated
    bytes would miss: a rename with identical content, an empty file appearing or disappearing,
    and any reordering.
    """

    files: tuple[SourceFile, ...] = ()
    content_hash: str = ''
    #: The suffix set the walk ADMITTED — the inventory's SCOPE, recorded because without it a
    #: comparison cannot tell drift from a scope difference. A baseline taken with
    #: ``--suffixes html`` re-hashed at the default scope reports every ``.html`` page as
    #: ``removed`` with none of them touched, and that verdict reaches the customer-facing
    #: run-integrity section as DRIFTED. Empty means the scope was not recorded (an inventory
    #: built before this field, or assembled by hand) — which is "unknown", never "the default".
    suffixes: tuple[str, ...] = ()
    #: What the AGENT sees of these bytes, when the scan could determine it — see
    #: :class:`VisibleProjection`. ``None`` means the scan did not record a projection, which is
    #: "unknown", never "the agent sees the raw bytes": a fidelity judge that assumes the raw
    #: source would grade a URL the compiler strips as ground truth the agent knew.
    visible_projection: VisibleProjection | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'content_hash': self.content_hash,
            'file_count': len(self.files),
            'files': [f.as_dict() for f in self.files],
            'suffixes': list(self.suffixes),
        }
        # ABSENT, not null, when unknown — the same posture ``render_actions`` takes on the
        # contract: a consumer must be able to tell "nobody recorded a projection" from "recorded,
        # and it withholds nothing" by the key alone.
        if self.visible_projection is not None:
            payload['visible_projection'] = self.visible_projection.as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceInventory:
        files = tuple(
            SourceFile(path=str(f.get('path', '')), sha256=str(f.get('sha256', '')), byte_count=int(f.get('bytes', 0) or 0))
            for f in (payload.get('files') or [])
            if isinstance(f, Mapping)
        )
        recorded_suffixes = payload.get('suffixes')
        projection = payload.get('visible_projection')
        return cls(
            files=files,
            content_hash=str(payload.get('content_hash', '') or ''),
            suffixes=normalize_suffixes(str(x) for x in recorded_suffixes) if isinstance(recorded_suffixes, (list, tuple)) else (),
            visible_projection=VisibleProjection.from_dict(projection) if isinstance(projection, Mapping) else None,
        )


@dataclass(frozen=True)
class DriftReport:
    """Recorded-vs-current source comparison. ADVISORY by design — see :func:`check_source_drift`."""

    matched: bool
    recorded_hash: str
    current_hash: str
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    #: Set when the comparison could not be made at all (no recorded inventory). Distinct from
    #: ``matched=False``: "nothing to compare against" is not "the sources drifted".
    blocked_reason: str | None = None

    def summary(self) -> str:
        if self.blocked_reason:
            return f'corpus source drift NOT checked: {self.blocked_reason}'
        if self.matched:
            # No counts: a matched inventory has empty added/removed/changed by construction,
            # and printing '0 difference(s)' would leave the reader wondering what was compared.
            return f'corpus sources match the recorded scan ({self.recorded_hash[:12]})'
        parts = []
        if self.added:
            parts.append(f'{len(self.added)} added ({", ".join(self.added[:3])}{" …" if len(self.added) > 3 else ""})')
        if self.removed:
            parts.append(f'{len(self.removed)} removed ({", ".join(self.removed[:3])}{" …" if len(self.removed) > 3 else ""})')
        if self.changed:
            parts.append(f'{len(self.changed)} changed ({", ".join(self.changed[:3])}{" …" if len(self.changed) > 3 else ""})')
        detail = '; '.join(parts) or 'hash differs, per-file digests unavailable'
        return f'corpus sources DRIFTED from the recorded scan ({self.recorded_hash[:12]} -> {self.current_hash[:12]}): {detail}'

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'matched': self.matched,
            'recorded_hash': self.recorded_hash,
            'current_hash': self.current_hash,
            'added': list(self.added),
            'removed': list(self.removed),
            'changed': list(self.changed),
            'summary': self.summary(),
        }
        if self.blocked_reason:
            out['blocked_reason'] = self.blocked_reason
        return out


def normalize_suffixes(values: Iterable[str]) -> tuple[str, ...]:
    """``('html', '.CSV')`` -> ``('.html', '.csv')`` — the one spelling rule for a suffix.

    An operator types ``--suffixes html``; the walk compares against the lower-cased name's dotted
    ending (:func:`_admits`). Without this the flag parses, runs, admits NOTHING, and the scan
    reports the same clean result as one that read the files — the exact silent miss the flag
    exists to close. De-duplicated, order preserved. ``json.gz`` spells ``.json.gz``.
    """
    return tuple(dict.fromkeys(f'.{s.strip().lstrip(".").lower()}' for s in values if s.strip().strip('.')))


def _admits(name: str, admitted: set[str]) -> bool:
    """Whether a file NAME ends with an admitted suffix — compound ones (``.json.gz``) included.

    ``Path(name).suffix`` sees only ``.gz``, so ``x.json.gz`` could never be admitted by the scope
    that names it. A name that IS a suffix (a dotfile called ``.md``) has no stem and stays out.
    """
    lowered = name.lower()
    return any(lowered.endswith(s) and lowered != s for s in admitted)


class SourceReader(Protocol):
    """How a scan turns one admitted file into searchable text — the seam an SDK plugs conversion into.

    This module stays reader-free: it owns no format table and no converter. The default reproduces
    a plain-text scan exactly; a reader that knows PDFs or archives is passed in by the caller that
    owns those converters.
    """

    def members(self, path: Path) -> Iterable[tuple[str, bytes]]:
        """``(member, raw)`` per addressable unit of ``path``: ``('', raw)`` for the file itself,
        ``('!/<member>', raw)`` per archive member. ``raw`` is what is HASHED — the bytes as stored,
        so a converter upgrade never reads as corpus drift. The unit id is ``<file id><member>``."""
        ...

    def decode(self, name: str, raw: bytes) -> tuple[str, bool, str]:
        """``(text, undecodable, warning)`` for one member, as :func:`decode_source` returns it.
        ``name`` is the unit id (``docs/a.zip!/d/m.txt``), whose suffix picks the reader."""
        ...


class _PlainReader:
    """The default :class:`SourceReader`: every file is one unit, decoded as text."""

    def members(self, path: Path) -> Iterable[tuple[str, bytes]]:
        return (('', path.read_bytes()),)

    def decode(self, name: str, raw: bytes) -> tuple[str, bool, str]:  # noqa: ARG002 - the protocol passes the name for readers that dispatch on it
        return decode_source(raw)


_PLAIN_READER: SourceReader = _PlainReader()


def _admitted_members(path: Path, rel: str, reader: SourceReader, admitted: set[str], skipped_suffixes: dict[str, int] | None) -> Iterator[tuple[str, bytes]]:
    """``(unit id, raw)`` per member of ``path`` the scope admits; the rest are counted like skipped files.

    The file itself was admitted by the walk. An archive member is admitted by the same suffix
    rule on its OWN name, so ``.zip`` in scope does not smuggle in every binary the archive holds.
    """
    for member, raw in reader.members(path):
        if member:
            name = Path(member).name
            if not _admits(name, admitted):
                if skipped_suffixes is not None and (suffix := Path(name).suffix.lower()):
                    skipped_suffixes[suffix] = skipped_suffixes.get(suffix, 0) + 1
                continue
        yield f'{rel}{member}', raw


def _iter_source_paths(root: Path, suffixes: Sequence[str], skipped_dirs: set[str], skipped_suffixes: dict[str, int] | None = None) -> Iterator[Path]:
    """Admissible files under ``root``, recording every directory — and every SUFFIX — whose skip COST something.

    ``os.walk`` with in-place ``dirnames`` pruning, not ``rglob('*')``: the glob descends fully
    into ``.git`` / ``.venv`` / ``node_modules`` and stats every entry before anything is
    discarded, which is far slower on a repo-sized root. Suffix is tested on the NAME before any
    stat, for the same reason.

    The noise filter drops those directories so a hash does not move every time a virtualenv is
    refreshed. But a dotted directory can also hold real documentation, and dropping it silently
    makes a partially-read corpus render exactly like a fully-read one. So a skipped directory is
    recorded IF it holds a file this scan would otherwise have admitted — probed lazily, stopping
    at the first hit — and stays quiet when it does not.

    ``skipped_suffixes`` is the same argument one level down, for the suffix filter: a file dropped
    by SUFFIX would otherwise be invisible. Counted per extension rather than per file, so a
    corpus root that is also a source tree reports one line per extension instead of three
    thousand. Extensionless files are not counted — ``--suffixes`` cannot name them, so a warning
    about them is a warning nobody can act on.
    """
    admitted = {s.lower() for s in suffixes}
    if root.is_file():
        # A FILE root is a one-file scope: one file of a shared directory, and nothing beside it.
        if _admits(root.name, admitted):
            yield root
        elif skipped_suffixes is not None and root.suffix:
            skipped_suffixes[root.suffix.lower()] = 1
        return
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        pruned = [d for d in dirnames if d in _SKIP_DIRS or d.startswith('.')]
        dirnames[:] = sorted(d for d in dirnames if d not in pruned)
        for name in pruned:
            if _holds_admissible(here / name, admitted):
                skipped_dirs.add((here / name).relative_to(root).as_posix())
        for name in filenames:
            if _admits(name, admitted):
                if (here / name).is_file():
                    found.append(here / name)
            elif skipped_suffixes is not None and (suffix := Path(name).suffix.lower()):
                skipped_suffixes[suffix] = skipped_suffixes.get(suffix, 0) + 1
    yield from sorted(found)


def skipped_suffix_warnings(skipped: Mapping[str, int], suffixes: Sequence[str]) -> list[str]:
    """One warning per extension this scan did not read, loudest first.

    THE half that closes the silent miss even for a customer who never passes ``--suffixes``.
    ``load_corpus`` already warns about a skipped DIRECTORY that held a readable source; a file
    dropped by suffix without a warning would leave a corpus of one ``README.md`` beside many
    ``.html`` help pages loading one unit and reporting exactly like a corpus of one file, with
    every ``kind: absence`` fact verified over that one unit.

    Names what was NOT read and what the scan reads, because those two together are the whole
    action. Widening is offered only for PLAIN TEXT in an unusual extension: the scan reads exactly
    the suffixes it was given, so a ``.pdf`` admitted by ``--suffixes`` is not read either — it is
    refused as binary (:func:`decode_source`). A format with no reader has to be converted.
    """
    ordered = sorted(skipped.items(), key=lambda pair: (-pair[1], pair[0]))
    reads = ', '.join(suffixes)
    out = [
        f'{count} file(s) with suffix {suffix} were NOT read: this scan has no reader for {suffix} (it reads {reads}). '
        f'Every fact was decided without them. Convert them, or, if they are plain text, pass --suffixes {suffix.lstrip(".")}'
        for suffix, count in ordered[:_MAX_SKIPPED_SUFFIX_WARNINGS]
    ]
    if len(ordered) > _MAX_SKIPPED_SUFFIX_WARNINGS:
        rest = ordered[_MAX_SKIPPED_SUFFIX_WARNINGS:]
        out.append(
            f'{sum(count for _, count in rest)} further file(s) across {len(rest)} other suffix(es) were NOT read '
            f'({", ".join(suffix for suffix, _ in rest[:10])}{" …" if len(rest) > 10 else ""})'
        )
    return out


def _holds_admissible(root: Path, admitted: set[str]) -> bool:
    """Whether a pruned directory holds any file this scan would have read. Short-circuits."""
    return any(any(_admits(name, admitted) for name in filenames) for _, _, filenames in os.walk(root))


def _flatten(value: Any, prefix: str, out: dict[str, str], oversized: list[str]) -> None:
    """Structured document -> dotted-path scalars, recording the scalars it had to drop.

    Lists index by position, which is what makes a record's field addressable
    (``assets.0.link_url``) without a schema. ``oversized`` collects the paths excluded by the
    length cap so the caller can warn: a dropped field shrinks the population a field-scoped
    fact is decided over, and an absence claim over a shrunken population reads as verified.
    """
    if isinstance(value, Mapping):
        for key, sub in value.items():
            _flatten(sub, f'{prefix}.{key}' if prefix else str(key), out, oversized)
    elif isinstance(value, (list, tuple)):
        for i, sub in enumerate(value):
            _flatten(sub, f'{prefix}.{i}' if prefix else str(i), out, oversized)
    elif value is not None and prefix:
        text = str(value)
        if len(text) <= _MAX_FIELD_CHARS:
            out[prefix] = text
        else:
            oversized.append(prefix)


def _structured_fields(path: Path, text: str) -> tuple[dict[str, str], str | None]:
    """``(fields, degradation_reason)`` for a structured source.

    ``reason`` is ``None`` when the source parsed AND yielded addressable scalars. It is a
    sentence otherwise — and the four ways that happens are different facts, not one "did not
    parse": the document raised, it parsed to nothing, it parsed to a top-level scalar with no
    addressable path, or every scalar exceeded the length cap. A reader chasing a syntax error
    that is not there is the cost of collapsing them.

    A degraded structured file is NOT an error: its ``text`` is still scanned, so the checks see
    its content — they just cannot scope themselves to a field inside it.
    """
    try:
        doc = json.loads(text) if path.suffix.lower() == '.json' else safe_yaml_load(text)
    except Exception as exc:  # noqa: BLE001 - any malformed structured source degrades to text-only
        # The diagnosis is the actionable half (a YAML line number, a JSON offset); discarding it
        # leaves a warning nobody can act on.
        return {}, f'did not parse ({type(exc).__name__}: {" ".join(str(exc).split())[:160]})'
    out: dict[str, str] = {}
    oversized: list[str] = []
    _flatten(doc, '', out, oversized)
    if out:
        return out, (
            f'{len(oversized)} field(s) exceeded the {_MAX_FIELD_CHARS}-char cap and are NOT field-addressable: {sorted(oversized)[:5]}' if oversized else None
        )
    if oversized:
        return out, f'every scalar exceeded the {_MAX_FIELD_CHARS}-char cap, so no field is addressable'
    return out, 'parsed cleanly but carries no addressable scalar (an empty document, or a top-level scalar)'


def decode_source(raw: bytes) -> tuple[str, bool, str]:
    """``(text, undecodable, warning)`` for one source's bytes.

    Three outcomes, and the third is why this is not a two-line try/except. Clean UTF-8 decodes
    and nothing is said. A file with a few bad bytes decodes lossily and is WARNED about — the
    damage is local, every check over the rest is sound. A file written wholly in another encoding
    (windows-1255, for example) decodes to a page of U+FFFD, and a literal matcher
    over that page finds none of the terms it was given: an absence claim then verifies
    VACUOUSLY, which is the one direction of this check that is silent and the one that mints
    adversarial probes. Such a unit is flagged ``undecodable`` so a consumer can refuse to
    decide over it, instead of being handed a warning nobody wired to a verdict.

    A binary CONTAINER (:data:`_BINARY_MAGIC`) is refused by name before any decode, and its text
    is empty: its bytes are syntax, not the document, whatever share of them happens to be ASCII.
    A caller with a reader for it converts first and never hands this function the container.
    """
    for magic, kind in _BINARY_MAGIC:
        if raw.startswith(magic):
            return (
                '',
                True,
                f'binary {kind} content, no reader here — its bytes are container syntax, not the document text, so NOTHING '
                'can be decided over it. Convert it to text, or scan with a reader for it',
            )
    try:
        return raw.decode('utf-8'), False, ''
    except UnicodeDecodeError:
        text = raw.decode('utf-8', errors='replace')
    replaced = text.count('\ufffd')
    if text and replaced / len(text) >= UNDECODABLE_REPLACEMENT_RATIO:
        return (
            text,
            True,
            f'not valid UTF-8 and {replaced} of {len(text)} character(s) decoded to U+FFFD — the text is destroyed, not '
            'merely damaged (a source written in another encoding, e.g. windows-1255). NOTHING can be decided over it: '
            'a term written in that encoding cannot match, so an absence claim would verify vacuously. Re-save the '
            'source as UTF-8',
        )
    return text, False, 'not valid UTF-8; decoded with replacement characters, so literal matches inside the damaged spans may not fire'


def load_corpus(
    root: Path, *, suffixes: Sequence[str] = DEFAULT_SOURCE_SUFFIXES, reader: SourceReader | None = None
) -> tuple[list[CorpusUnit], SourceInventory, list[str]]:
    """Walk ``root`` into units + a source inventory + warnings.

    Returns ``(units, inventory, warnings)``. Warnings name every source admitted in a DEGRADED
    form (unreadable bytes, unparseable structure), every directory whose skip cost a readable
    source, AND every EXTENSION this scan did not admit — a scan that silently dropped a source
    reports the same clean result as one that read it and found nothing.

    ``reader`` (:class:`SourceReader`) splits and converts each admitted file; ``None`` is the
    plain-text scan.
    """
    reader = reader or _PLAIN_READER
    admitted = {s.lower() for s in suffixes}
    units: list[CorpusUnit] = []
    files: list[SourceFile] = []
    warnings: list[str] = []
    skipped_dirs: set[str] = set()
    skipped_suffixes: dict[str, int] = {}
    for path in _iter_source_paths(root, suffixes, skipped_dirs, skipped_suffixes):
        for uid, raw in _admitted_members(path, _source_id(root, path), reader, admitted, skipped_suffixes):
            files.append(SourceFile(path=uid, sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw)))
            text, undecodable, decode_warning = reader.decode(uid, raw)
            if decode_warning:
                warnings.append(f'{uid}: {decode_warning}')
            fields: dict[str, str] = {}
            # The member's OWN name decides: ``a.zip!/d/rows.json`` is structured, ``x.json.gz`` is not.
            if Path(uid).suffix.lower() in _STRUCTURED_SUFFIXES:
                fields, degraded = _structured_fields(Path(uid), text)
                if degraded:
                    warnings.append(f'{uid}: {degraded}. It is searchable as TEXT; field-scoped facts address only what IS flattened')
            units.append(CorpusUnit(unit_id=uid, text=text, fields=fields, undecodable=undecodable))
    warnings.extend(
        f'{d}/: skipped as a build/VCS directory, but it holds file(s) this scan would otherwise have read — they are NOT in the corpus or its content hash'
        for d in sorted(skipped_dirs)
    )
    warnings.extend(skipped_suffix_warnings(skipped_suffixes, suffixes))
    return units, build_inventory(files, suffixes=suffixes), warnings


def build_source_inventory(roots: Sequence[Path], *, suffixes: Sequence[str] = DEFAULT_SOURCE_SUFFIXES, reader: SourceReader | None = None) -> SourceInventory:
    """The source fingerprint ALONE — walk, read, hash, stop.

    ``load_corpus`` additionally decodes and YAML-parses every structured source, which is where
    essentially all of its time goes (measured ~1369 ms of a 1377 ms load on a 7 MB corpus). A
    drift re-check reads none of that, so it does not pay for it. Shares ``_iter_source_paths``,
    the member split and the prefixing rule with ``load_corpus_roots``, so the two cannot disagree
    about which files are in scope or what a unit is called. ``reader.decode`` is never called.
    """
    reader = reader or _PLAIN_READER
    admitted = {s.lower() for s in suffixes}
    prefixed = len(roots) > 1
    files: list[SourceFile] = []
    for root in roots:
        skipped: set[str] = set()
        for path in _iter_source_paths(root, suffixes, skipped):
            rel = _source_id(root, path)
            rel = f'{root.name}/{rel}' if prefixed and not root.is_file() else rel
            for uid, raw in _admitted_members(path, rel, reader, admitted, None):
                files.append(SourceFile(path=uid, sha256=hashlib.sha256(raw).hexdigest(), byte_count=len(raw)))
    return build_inventory(files, suffixes=suffixes)


def build_inventory(files: Iterable[SourceFile], *, suffixes: Sequence[str] = ()) -> SourceInventory:
    """The per-file digests plus the one hash over them, in sorted path order.

    ``suffixes`` is the walk's SCOPE, stored on the inventory rather than left implicit: see
    :attr:`SourceInventory.suffixes`. It is NOT hashed — two scans of the same files agree on
    their content whatever admitted them, and folding the scope into the hash would report a
    widened ``--suffixes`` that found nothing new as content drift.
    """
    ordered = tuple(sorted(files, key=lambda f: f.path))
    digest = hashlib.sha256()
    for f in ordered:
        digest.update(f.path.encode('utf-8'))
        digest.update(b'\x00')
        digest.update(f.sha256.encode('ascii'))
        digest.update(b'\n')
    return SourceInventory(files=ordered, content_hash=digest.hexdigest(), suffixes=normalize_suffixes(suffixes))


def suffix_scope_mismatch(recorded: Sequence[str], current: Sequence[str]) -> str | None:
    """Why a comparison across THESE two suffix scopes would not measure drift, or ``None``.

    ONE owner for the sentence and for the decision, because three call sites ask it — the
    ``check`` report, the ``recheck`` command and the SDK's run-start preflight — and a scope
    rule spelled three times is the drift this package exists to stop.

    A scope difference is INDISTINGUISHABLE from drift by the file lists alone: a baseline
    recorded with ``--suffixes html`` and re-hashed at the default scope reports every ``.html``
    page as ``removed`` with not one of them touched. The removal is PARTIAL, so
    ``RecheckOutcome.shares_no_path`` — the guard that catches an entirely different root — never
    fires, and the manufactured verdict reaches the customer-facing run-integrity section as
    ``corpus sources DRIFTED``.

    An UNRECORDED scope on either side (the empty tuple: an inventory built before the field
    existed, or assembled by hand) is not a mismatch and must not be reported as one. It is
    unknown, and turning "I cannot tell" into an abstention here would take a comparison that IS
    measurable — every file digest, on both sides — off every baseline written before this field.
    The abstention is for a scope difference that has been SEEN, never for one that cannot be
    ruled out.
    """
    was, now = set(normalize_suffixes(recorded)), set(normalize_suffixes(current))
    if not was or not now or was == now:
        return None
    only_baseline = ', '.join(sorted(was - now))
    only_now = ', '.join(sorted(now - was))
    admitted = f'the baseline admitted {only_baseline} and this check did not' if only_baseline else ''
    widened = f'this check admits {only_now} and the baseline did not' if only_now else ''
    difference = '; '.join(part for part in (admitted, widened) if part)
    return (
        f'the two sides were walked at DIFFERENT scopes — {difference}. A file outside the narrower scope reports as '
        f'added or removed with nothing having changed on disk, so drift was NOT measured. Re-check with the suffix '
        f'set the baseline recorded ({", ".join(sorted(was))}), or re-record the baseline at this scope'
    )


def check_source_drift(recorded: SourceInventory | None, current: SourceInventory) -> DriftReport:
    """Compare a recorded inventory against the sources as they are now.

    ADVISORY on purpose, and this is a scoping decision, not an oversight: a customer editing
    their own documentation between the scan and the run is NORMAL, and hard-failing on it
    would make the check the first thing anyone disables. What it must never do is stay silent
    — the report carries the drift either way, so a suite scored against a moved corpus says so.

    ABSTAINS on a scope mismatch (:func:`suffix_scope_mismatch`) rather than reporting one: two
    inventories walked at different suffix scopes differ by files nobody touched, and "drifted"
    is a claim about the CORPUS, not about the walk.
    """
    if recorded is not None and (scope := suffix_scope_mismatch(recorded.suffixes, current.suffixes)) is not None:
        return DriftReport(matched=False, recorded_hash=recorded.content_hash, current_hash=current.content_hash, blocked_reason=scope)
    if recorded is None or not recorded.content_hash:
        return DriftReport(
            matched=False,
            recorded_hash='',
            current_hash=current.content_hash,
            blocked_reason='no source inventory was recorded at scan time, so there is nothing to compare the current sources against',
        )
    if recorded.content_hash == current.content_hash:
        return DriftReport(matched=True, recorded_hash=recorded.content_hash, current_hash=current.content_hash)
    was = {f.path: f.sha256 for f in recorded.files}
    now = {f.path: f.sha256 for f in current.files}
    return DriftReport(
        matched=False,
        recorded_hash=recorded.content_hash,
        current_hash=current.content_hash,
        added=tuple(sorted(now.keys() - was.keys())),
        removed=tuple(sorted(was.keys() - now.keys())),
        changed=tuple(sorted(p for p in was.keys() & now.keys() if was[p] != now[p])),
    )


@dataclass(frozen=True)
class EmptyRoot:
    """A declared corpus root that admitted no file at this scan's suffix scope, with what it DID hold.

    Reported, never refused: the rest of the corpus is still read and decided, and the report holds
    generation on this root (``scan.corpus_root_empty``) until the scope is widened or the root dropped.
    """

    root: Path
    admitted_suffixes: tuple[str, ...]
    #: ``{suffix: file count}`` of what the walk skipped by suffix under this root — what ``--suffixes`` could admit.
    skipped_by_suffix: dict[str, int]


def _empty_root(root: Path, suffixes: Sequence[str], reader: SourceReader | None = None) -> EmptyRoot:
    """The census of a root that admitted nothing. Walks it once more; a file it yields is an archive
    whose members were all skipped, so only its member split is read — to count them."""
    reader = reader or _PLAIN_READER
    admitted = {s.lower() for s in suffixes}
    skipped: dict[str, int] = {}
    for path in _iter_source_paths(root, suffixes, set(), skipped):
        for _ in _admitted_members(path, '', reader, admitted, skipped):
            pass
    return EmptyRoot(root=root, admitted_suffixes=normalize_suffixes(suffixes), skipped_by_suffix=dict(sorted(skipped.items())))


def load_corpus_roots(
    roots: Sequence[Path], *, suffixes: Sequence[str] = DEFAULT_SOURCE_SUFFIXES, reader: SourceReader | None = None
) -> tuple[list[CorpusUnit], SourceInventory, list[str], list[EmptyRoot]]:
    """Load several corpus roots into ONE unit list + ONE inventory, and name every root that admitted nothing.

    Returns ``(units, inventory, warnings, empty_roots)``. A root that admits no file is RETURNED,
    not raised: a contract over it decides nothing about the facts scoped to it, which the report
    holds on, while every other root is still read and decided.

    Unit ids are prefixed with the root's own directory name when more than one root is given,
    so ``docs/x.md`` under two roots stays two addressable units. With a single root the ids are
    left bare — prefixing them would change every fact's scope glob for no gain.

    ``suffixes`` is FORWARDED to ``load_corpus``: this is the only entry point either CLI uses, so
    it is how a command line asks for the widening. A caller that wants ``.html`` help pages in
    scope passes them here.
    ``reader`` is forwarded the same way, to every root.
    """
    if len(roots) == 1:
        units, inventory, warnings = load_corpus(roots[0], suffixes=suffixes, reader=reader)
        return units, inventory, warnings, ([] if units else [_empty_root(roots[0], suffixes, reader)])
    units: list[CorpusUnit] = []
    empty: list[EmptyRoot] = []
    files: list[SourceFile] = []
    warnings: list[str] = []
    seen: dict[str, Path] = {}
    for root in roots:
        prefix = root.name
        if (clash := seen.get(prefix)) is not None:
            raise ValueError(
                f'two corpus roots share the directory name {prefix!r} ({clash} and {root}), so their unit ids would '
                'collide and one would silently shadow the other. Pass roots with distinct names, or one common parent.'
            )
        seen[prefix] = root
        sub_units, sub_inventory, sub_warnings = load_corpus(root, suffixes=suffixes, reader=reader)
        if not sub_units:
            empty.append(_empty_root(root, suffixes, reader))
        # A file root's one unit is already named by the file; prefixing it would repeat the name.
        unit_prefix = '' if root.is_file() else f'{prefix}/'
        units.extend(CorpusUnit(unit_id=f'{unit_prefix}{u.unit_id}', text=u.text, fields=u.fields, undecodable=u.undecodable) for u in sub_units)
        files.extend(SourceFile(path=f'{unit_prefix}{f.path}', sha256=f.sha256, byte_count=f.byte_count) for f in sub_inventory.files)
        warnings.extend(f'{unit_prefix}{w}' for w in sub_warnings)
    return units, build_inventory(files, suffixes=suffixes), warnings, empty


def _source_id(root: Path, path: Path) -> str:
    """What a source is called under its root: the path relative to a directory root, or the name of a file root."""
    return path.name if root.is_file() else path.relative_to(root).as_posix()
