"""Resolve every typed fact's ``source`` against the files the scan actually read.

:mod:`plurai.scan_contract.facts` decides a fact against the CORPUS — the customer's knowledge
documents. It never looks at the citation the author wrote beside the claim, and that citation is
where the cheap half of a PRD's defects live: a fact citing a line a few lines away from its
symbol, a count of source files that disagrees with the read sites, a claim of no external
services beside a loaded ``.env`` that enables one. Each is findable by reading a file the scan
already has open; found after generation, each costs a scenario regeneration.

THE ``.env`` CASE IS NOT THIS MODULE'S. Resolving a citation
proves the ``.env`` exists and carries the fact's subject — it never reads the file for a service
that is switched ON, and for an ABSENCE claim a ``resolved`` verdict is the perverse case: the
subject being present on the cited line is evidence AGAINST the claim, which this check has no
vocabulary to say. :mod:`plurai.scan_contract.services` owns that question and answers it
against the same roots.

Two questions, and the second is the one a plain existence test misses:

* does the citation RESOLVE — is there a file or directory there at all;
* does the cited LINE support the claim — a ``path:LINE`` citation eleven lines off resolves
  perfectly and cites nothing. So the line is checked for carrying the fact's own ``subject``,
  and when the subject is elsewhere in the file the verdict names the line it is actually on.

NOTHING here holds generation. Every verdict is reported — as an advisory item, with the whole
audit on the artifact's ``fact_citations`` section — because citations are provenance, and a
±3-line window over a substring match is a heuristic. ONE line-level answer is a DEFECT the author
can correct: the subject is IN the file and NOT where the citation says, and the verdict names the
line they meant. The others are not even that — "the subject is nowhere in
this file" is the CORRECT citation for every absence-shaped predicate in the vocabulary
(``undocumented_to_model``, ``omits``, ``rejects``, ``absent_from_source``), and "the line could
not be read" is a statement about this checker rather than about the PRD.

Deliberately NOT a re-decision of the claim against the cited document. A ``data_profile.md``
is a document ABOUT the fields it profiles, so running an absence fact's terms over it reports
the field name it was written to discuss — a contradiction manufactured by the category error,
not found. The corpus check is where a claim is decided; this one only establishes that the
evidence the author cited exists and says what they said it says.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plurai.scan_contract.corpus import decode_source
from plurai.scan_contract.facts import KIND_ASSERTED, PREDICATE_INSTRUCTS, ScanFact

#: The citation resolved: the path exists, and a line citation lands on the subject.
CITATION_RESOLVED = 'resolved'
#: Nothing is there. The author cited a file this machine does not have.
CITATION_MISSING = 'missing'
#: The path exists, the subject is IN it, and the cited line is not where it is — the off-by-N
#: miss this check exists for. Also a line past the end of the file, and a line number on a
#: directory. The one line-level DEFECT: each is a citation the author can re-derive and correct,
#: because this verdict names the line they meant.
CITATION_MISPLACED = 'misplaced'
#: The fact carries no ``source`` at all.
CITATION_UNCITED = 'uncited'
#: A citation this checker deliberately does not resolve — ``corpus:<doc>`` (addressed by the
#: corpus check's own scope globs, not by a filesystem path) or free prose. Recorded, never passed.
CITATION_NOT_A_PATH = 'not_a_path'
#: The path resolved and the LINE could not be checked AT ALL: an unreadable or undecodable file,
#: a fact declaring no locatable ``subject``, a line reference in a form this checker cannot
#: parse. NEITHER a defect NOR resolved, and that is the point — counting it resolved asserts a
#: line was checked that was not, and calling it a defect blames the PRD for this checker's reach.
CITATION_UNCHECKED = 'unchecked'
#: The line was checked and the subject occurs NOWHERE in the cited file. NOT a defect, because
#: the closed predicate vocabulary is full of absence-shaped claims — ``undocumented_to_model``,
#: ``omits``, ``rejects``, ``absent_from_source`` — whose whole content is that the subject is not
#: there, so the citation naming where it is not is CORRECT. Reported, never counted as resolved.
CITATION_UNSUPPORTED = 'unsupported'

#: The statuses that name a citation DEFECT the author can correct (:attr:`CitationAudit.defective`).
CITATION_DEFECTS: tuple[str, ...] = (CITATION_MISSING, CITATION_MISPLACED, CITATION_UNCITED)

#: How far from the cited line the subject may sit and still count as cited. Three lines covers a
#: decorator, a wrapped signature and a docstring opener — the formatting drift a correct citation
#: legitimately accumulates. A wider window would admit a citation that is several lines off,
#: which this check exists to catch.
CITATION_WINDOW = 3

#: Shortest subject token worth locating. Below this a match is noise (``id``, ``db``) and the
#: support check would pass on any file, which reads as coverage.
_MIN_SUBJECT_TOKEN = 3

#: Most bytes read out of a cited file to check one line. A cited 200 MB artifact decoded whole
#: is a scan that dies on memory instead of reporting a finding; past this the LINE is simply
#: unchecked, and says so.
_MAX_CITED_BYTES = 4 * 1024 * 1024

#: ``<path>:<line>``, plus the forms an LLM-authored PRD actually produces for the same thing: a
#: RANGE (``agent.py:41-42``), a GitHub anchor (``agent.py#L41``), and ``path:line:col``. Only the
#: FIRST number is read — a range's first line is where the claim was read. Accepting them is not
#: cosmetic: without them each one falls through to a whole-string path lookup, resolves to nothing, and
#: blocks the build on a citation that is merely spelled differently. The path may not contain
#: whitespace, a colon or a ``#``, which keeps a Windows drive letter and a ``corpus:`` prefix out.
_LINE_CITATION = re.compile(r'^(?P<path>[^\s:#]+)(?::|#L)(?P<line>\d+)(?:-L?(?P<end>\d+))?(?:[-:]L?\d+)*$')

#: Where a path stops and a line reference this checker could not parse begins. Used only to tell
#: "the file is not there" from "the file is there and I cannot read the suffix".
_LINE_SUFFIX = re.compile(r'[:#]')

#: A web citation: a URL scheme, or a scheme-less ``www.`` host. Never resolved against the tree.
_URL_CITATION = re.compile(r'^(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)', re.IGNORECASE)

#: The agent-facts vocabulary's token for "read off the knowledge corpus", handled before the
#: path branch because ``corpus:products.md`` is a unit id and not a file on this machine.
_CORPUS_PREFIX = 'corpus:'


@dataclass(frozen=True)
class CitationVerdict:
    """What the citation on ONE fact turned out to be."""

    fact_id: str
    source: str
    status: str
    reason: str
    #: The path this resolved to, relative to the root that held it. Empty when nothing resolved.
    resolved_path: str = ''
    #: Whether the cited LINE was actually examined for the subject. Separate from the status
    #: because three different statuses come out of a line that WAS read, and three more out of
    #: one that was not — and "no line citation could be checked at all" is the fact that decides
    #: whether this check ran or was blocked. A ``subject`` absent from every fact turns the
    #: off-by-N check off wholesale while every verdict still reads normal.
    line_checked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            'fact_id': self.fact_id,
            'source': self.source,
            'status': self.status,
            'reason': self.reason,
            'resolved_path': self.resolved_path,
            'line_checked': self.line_checked,
        }


@dataclass(frozen=True)
class CitationAudit:
    """Every fact's citation verdict, plus whether the audit could run at all.

    ``checked`` is separate from an empty ``verdicts`` list for the reason every other check here
    keeps the distinction: "no citation was wrong" and "no root was supplied to resolve against"
    are the same shape and license very different confidence.
    """

    checked: bool
    blocked_reason: str = ''
    verdicts: tuple[CitationVerdict, ...] = ()
    #: The roots citations were resolved against, as given.
    roots: tuple[str, ...] = ()

    @property
    def defective(self) -> list[CitationVerdict]:
        """Citations that point at evidence that is not there: nothing at the path, a line the subject is not
        on, or no source at all. Advisory — reported, never a hold."""
        return [v for v in self.verdicts if v.status in CITATION_DEFECTS]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for verdict in self.verdicts:
            out[verdict.status] = out.get(verdict.status, 0) + 1
        return out

    @property
    def line_support_established(self) -> bool:
        """Whether ANY cited line was actually read for its subject.

        The question the ``resolved`` count cannot answer. ``subject`` is what a ``path:LINE``
        citation is checked against, and a facts document written before it existed — or by hand,
        or round-tripped through an older artifact — carries none, so every line citation is
        UNCHECKED while the summary still reads like a clean audit. That is a check that did not
        run, and the ledger has to say so.
        """
        return any(v.line_checked for v in self.verdicts)

    def unestablished_reason(self) -> str:
        """Empty when this audit decided something; otherwise WHY it decided nothing.

        Read by the report to file this check ``blocked`` rather than ``ran``. A defective verdict
        is always something decided, so this is silent whenever there is one.
        """
        if self.defective:
            return ''
        unchecked = [v for v in self.verdicts if v.status == CITATION_UNCHECKED]
        if unchecked and not self.line_support_established:
            return (
                f'not one cited LINE could be checked ({len(unchecked)} of {len(self.verdicts)} citation(s)): '
                + ', '.join(f'{v.fact_id} — {v.reason}' for v in unchecked[:3])
                + '. Path existence was resolved; the off-by-N line check this exists for did NOT run on any fact'
            )
        if not any(v.status == CITATION_RESOLVED for v in self.verdicts):
            return (
                f'none of the {len(self.verdicts)} declared citation(s) resolved to anything this check could decide — '
                'they are corpus ids, prose, or references it could not read. Nothing was refuted and nothing was established'
            )
        return ''

    def summary(self) -> str:
        if not self.checked:
            return f'fact citations NOT resolved: {self.blocked_reason}'
        counts = self.counts()
        resolved = counts.get(CITATION_RESOLVED, 0)
        lines_checked = sum(1 for v in self.verdicts if v.line_checked)
        return (
            f'{resolved} of {len(self.verdicts)} citation(s) resolved against {len(self.roots)} root(s), '
            f'{lines_checked} with the cited LINE read for its subject; '
            + (', '.join(f'{n} {status}' for status, n in sorted(counts.items()) if status != CITATION_RESOLVED) or 'nothing else to report')
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'checked': self.checked,
            'blocked_reason': self.blocked_reason,
            'roots': list(self.roots),
            'counts': self.counts(),
            'summary': self.summary(),
            'verdicts': [v.as_dict() for v in self.verdicts],
        }


def _looks_like_a_path(token: str) -> bool:
    """Whether ``token`` is meant as a filesystem citation rather than prose.

    A path has no whitespace and either a directory separator or a suffix. The alternative —
    trying every source as a path — files a MISSING finding against every fact whose author wrote
    a sentence, which is the noise that gets a check switched off. A URL (``scheme://…`` or
    ``www.…``) is a citation of a web page, not of the scanned tree, so it is not a path either.
    """
    if not token or any(ch.isspace() for ch in token) or _URL_CITATION.match(token):
        return False
    return '/' in token or Path(token).suffix != ''


def _resolve(path: str, roots: Sequence[Path]) -> Path | None:
    """The first root under which ``path`` exists, or ``None``.

    Refuses an absolute path and any traversal out of the root: a citation is a claim about the
    scanned tree, and one that resolves to ``/etc/passwd`` has not established anything about it.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return None
    for root in roots:
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root.resolve()):
            continue
        if resolved.exists():
            return resolved
    return None


def _subject_token(fact: ScanFact) -> str:
    """The leaf of the fact's subject — the identifier a line citation should carry.

    ``compare_products.attributes`` cites the argument, so the leaf is what appears on the line;
    the dotted parent is the tool, which appears on a different one.
    """
    leaf = fact.subject.strip().rsplit('.', 1)[-1]
    return leaf if len(leaf) >= _MIN_SUBJECT_TOKEN else ''


def read_source_file(resolved: Path) -> tuple[str, str]:
    """One scanned file's text, or the REASON it could not be read. Never raises.

    Shared with :mod:`plurai.scan_contract.services`, which reads the same customer tree and owes
    the same promise; a second reader would be a second set of guards to keep in step.

    A cited file this process cannot read is a FINDING, not a crash. Unguarded, one mode-000
    file — or a FIFO inside the source root, or an I/O error — would take down fact verification, the
    URL allowlist, the source hash AND the drift baseline for the whole scan, with a traceback in
    place of a report and no artifact written at all.
    """
    if not resolved.is_file():
        # A socket, a FIFO, a device. `is_file` is what keeps the read below from blocking
        # forever on a named pipe the author happened to cite.
        return '', 'is not a regular file'
    try:
        with resolved.open('rb') as handle:
            raw = handle.read(_MAX_CITED_BYTES + 1)
    except OSError as exc:
        return '', f'could not be read ({type(exc).__name__}: {" ".join(str(exc).split())[:120]})'
    if len(raw) > _MAX_CITED_BYTES:
        return '', f'is larger than the {_MAX_CITED_BYTES // (1024 * 1024)} MB this check reads'
    text, undecodable, _warning = decode_source(raw)
    if undecodable:
        # The characters an identifier is made of are exactly what the decode destroyed, so a
        # miss here would be about the encoding, not about the citation.
        return '', 'did not decode as UTF-8'
    return text, ''


def _line_verdict(fact: ScanFact, resolved: Path, path: str, line: int) -> CitationVerdict:
    """Whether the cited LINE supports the claim, given the file resolved."""

    def verdict(status: str, reason: str, *, line_checked: bool = False) -> CitationVerdict:
        return CitationVerdict(fact_id=fact.fact_id, source=fact.source, status=status, reason=reason, resolved_path=path, line_checked=line_checked)

    if resolved.is_dir():
        return verdict(CITATION_MISPLACED, f'{path} is a directory, so the line number {line} addresses nothing')
    text, unreadable = read_source_file(resolved)
    if unreadable:
        return verdict(CITATION_UNCHECKED, f'{path} exists but {unreadable}, so line {line} was never read for the subject')
    lines = text.splitlines()
    if line < 1 or line > len(lines):
        return verdict(CITATION_MISPLACED, f'{path} has {len(lines)} line(s); the citation points at line {line}', line_checked=True)
    if fact.predicate == PREDICATE_INSTRUCTS:
        # A rule's subject is its own location, so the dotted leaf is no identifier. Whether the
        # quote sits on the line is the PRD lint's question (`lint_prd.check_rule_coverage`).
        subject = _LINE_CITATION.match(fact.subject.strip())
        cited = _LINE_CITATION.match(fact.source.strip())
        if subject is None or cited is None or subject.group('path', 'line', 'end') != cited.group('path', 'line', 'end'):
            return verdict(CITATION_MISPLACED, f'this rule fact names {fact.subject!r} as its location and cites {fact.source!r}; the two must be one place')
        return verdict(
            CITATION_RESOLVED, f'{path}:{line} is the location this rule fact names; the quote at that line is checked by the PRD lint (check_rule_coverage)'
        )
    token = _subject_token(fact)
    if not token:
        return verdict(
            CITATION_UNCHECKED,
            f'{path}:{line} exists, but the fact declares no `subject` long enough to locate ({fact.subject!r}), so the LINE was '
            'NOT checked — this citation is unresolved, not clean',
        )
    hits = [i for i, body in enumerate(lines, start=1) if token in body]
    if not hits:
        return verdict(
            CITATION_UNSUPPORTED,
            f'{path} carries no occurrence of {token!r} anywhere, so nothing on line {line} was shown to support this claim. '
            'Not a defect on its own — an absence-shaped claim cites exactly the place its subject is NOT — but nothing here '
            'establishes the citation either',
            line_checked=True,
        )
    if any(abs(hit - line) <= CITATION_WINDOW for hit in hits):
        return verdict(CITATION_RESOLVED, f'{path}:{line} carries {token!r}', line_checked=True)
    nearest = min(hits, key=lambda hit: abs(hit - line))
    return verdict(
        CITATION_MISPLACED,
        f'{path}:{line} does not carry {token!r}; the nearest occurrence is line {nearest}. Cite {path}:{nearest} — a claim '
        'read off a line that does not mention its subject is a claim nobody can re-derive',
        line_checked=True,
    )


def cited_lines(fact: ScanFact, roots: Sequence[Path]) -> str:
    """The text of the line(s) ``fact.source`` cites (``path:LINE`` or ``path:FROM-TO``), or ``''`` when it resolves to none."""
    match = _LINE_CITATION.match(fact.source.strip())
    resolved = _resolve(match.group('path'), roots) if match is not None else None
    if match is None or resolved is None or resolved.is_dir():
        return ''
    text, unreadable = read_source_file(resolved)
    first = int(match.group('line'))
    last = int(match.group('end') or first)
    return '' if unreadable else '\n'.join(text.splitlines()[first - 1 : last] if first >= 1 else [])


def audit_citation(fact: ScanFact, roots: Sequence[Path]) -> CitationVerdict:
    """Resolve ONE fact's citation."""

    def verdict(status: str, reason: str, resolved_path: str = '') -> CitationVerdict:
        return CitationVerdict(fact_id=fact.fact_id, source=fact.source, status=status, reason=reason, resolved_path=resolved_path)

    source = fact.source.strip()
    if not source:
        return verdict(
            CITATION_UNCITED,
            'the fact declares no source. It is read as a contract by the generator and by every judge, so a wrong one '
            'is authored into every scenario built on it — name the file, profile or probe report it was read from',
        )
    if source.lower().startswith(_CORPUS_PREFIX):
        # Which check owns it depends on the KIND. Saying "the corpus check decides that fact" of
        # an ``asserted`` fact is a deferral to a check that does not exist for it: the corpus
        # side answers UNVERIFIED_NO_STATIC_ROUTE, so both halves point at each other and nothing
        # ever decided the claim.
        owner = (
            'and nothing decides it — this fact declares no executable `check`, so the corpus side reports it '
            'asserted-unverified too. Give it a `check:` if reading the corpus settles the claim'
            if fact.kind == KIND_ASSERTED
            else "which this check does not resolve — the corpus check decides that fact against the corpus itself, through the fact's own scope"
        )
        return verdict(CITATION_NOT_A_PATH, f'{source} names a corpus document, {owner}')
    if not _looks_like_a_path(source):
        return verdict(CITATION_NOT_A_PATH, f'{source!r} is not a filesystem citation, so nothing here could resolve it')
    if (match := _LINE_CITATION.match(source)) is not None:
        path = match.group('path')
        resolved = _resolve(path, roots)
        if resolved is None:
            return verdict(CITATION_MISSING, f'{path} does not exist under any scanned root')
        return _line_verdict(fact, resolved, path, int(match.group('line')))
    resolved = _resolve(source, roots)
    if resolved is None:
        # …unless the FILE is there and only the line reference is in a form this checker cannot
        # parse. Reporting that as MISSING blames the PRD for this checker's own reach, on a
        # citation whose file the author can point at.
        head = _LINE_SUFFIX.split(source, maxsplit=1)[0]
        if head != source and _resolve(head, roots) is not None:
            return verdict(
                CITATION_UNCHECKED,
                f'{head} exists, but the line reference in {source!r} is not a form this check can read '
                '(it reads `path:LINE`, `path:FROM-TO`, `path:LINE:COL` and `path#LLINE`), so the LINE was not checked',
                resolved_path=head,
            )
        return verdict(CITATION_MISSING, f'{source} does not exist under any scanned root')
    return verdict(CITATION_RESOLVED, f'{source} exists', resolved_path=source)


def audit_citations(facts: Iterable[ScanFact], roots: Sequence[Path]) -> CitationAudit:
    """Resolve every fact's citation against ``roots``. Never raises — see :func:`read_source_file`,
    which is where the only I/O is and where that promise is kept.

    With no root the audit is BLOCKED rather than empty: a contract that reports "0 bad
    citations" because it was pointed at nothing is the coverage theatre this package removes.
    """
    declared = list(facts)
    if not roots:
        return CitationAudit(
            checked=False,
            blocked_reason='no source root was supplied, so no `source:` could be resolved — the citations are unchecked, not clean',
        )
    if not declared:
        return CitationAudit(checked=False, blocked_reason='no typed fact was declared for this scan, so there was no citation to resolve')
    return CitationAudit(checked=True, verdicts=tuple(audit_citation(fact, roots) for fact in declared), roots=tuple(str(r) for r in roots))
