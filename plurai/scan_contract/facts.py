"""Verify the typed scan facts THEMSELVES against the corpus, upstream of every consumer.

The scenario fact ledger (``scenario_facts``) validates SCENARIOS against the facts a
generator declared. This module validates the facts BEFORE generation. A false premise — "no
compliant answer exists for X", when the corpus plainly documents X — would otherwise propagate
into generation and suppress the scenarios that would have caught it.

Relationship to the scenario generator's own absence check: that one runs DURING generation,
over the grounding material a single scenario was given, and keys on the token ``absent``. It is
a repair-loop violation, not a terminal error, and it can only see what that scenario retrieved.
This module runs BEFORE any generation spends, over the whole scanned corpus, and holds
generation on what it refutes. They overlap deliberately and do not share state; if a
scan-verified absence ever projects INTO the ledger entry a generator authors, the ledger check
becomes a confirmation rather than a re-derivation. That projection is not built here.

A corpus-DERIVABLE fact is checked at scan time, and a contradicted fact HOLDS generation (the
``scan.fact_contradicted`` finding the report mints) with the fact and the contradicting corpus
unit both named, until the fact is re-authored or the finding overruled.

Two properties are load-bearing, and both are about honesty rather than coverage:

**Only derivable facts are checked.** Counts, absences, memberships and existence claims are
decidable against a corpus. A behavioural claim ("the agent should decline politely") is not.
Those are stamped :data:`UNVERIFIED_AT_SCAN` and counted separately, so nobody can read this
check as evidence about them. A fact whose KIND is derivable but which ships no typed payload
to check gets the same stamp with a pointed reason — never a quiet pass.

**Corpus outranks the PRD/brief.** The existing grounding-context contract already says the
customer's own knowledge base wins on knowledge facts. The contradiction message names BOTH
sides — the fact, its origin, and the corpus unit that disagrees — so the reader is not left to
guess which document to fix.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, get_args

from plurai.scan_contract._suite_findings import EXCERPT_CHARS as SUITE_EXCERPT_CHARS
from plurai.scan_contract._suite_findings import Severity
from plurai.scan_contract.corpus import CorpusUnit

#: Fact kinds this checker can DECIDE against a corpus.
KIND_ABSENCE = 'absence'
KIND_EXISTENCE = 'existence'
KIND_COUNT = 'count'
#: ...and the kind it deliberately cannot: a claim about behaviour, not about content.
KIND_BEHAVIOURAL = 'behavioural'
#: ...and the kind that says so about ITSELF: a claim its author declared with no executable
#: check. Projected rather than dropped, so the facts document never silently shrinks to the
#: checkable subset: a report over some of the declared facts must not render like a report over
#: all of them. It verifies NOTHING; it makes the PROCEED provisional.
KIND_ASSERTED = 'asserted'

#: The kinds ``verify_fact`` actually implements a branch for. Membership here is what licenses
#: a fact to fall past the guards into the per-kind dispatch, so a kind added here WITHOUT a
#: branch would inherit COUNT semantics and contradict everything against ``expected_count=None``
#: — a blocking verdict from a check nobody wrote. The final fallthrough in ``verify_fact``
#: catches that.
DERIVABLE_KINDS: frozenset[str] = frozenset({KIND_ABSENCE, KIND_EXISTENCE, KIND_COUNT})
FACT_KINDS: frozenset[str] = DERIVABLE_KINDS | {KIND_BEHAVIOURAL, KIND_ASSERTED}

FactStatus = Literal['verified', 'contradicted', 'unverified_at_scan']

VERIFIED: FactStatus = 'verified'
CONTRADICTED: FactStatus = 'contradicted'
UNVERIFIED_AT_SCAN: FactStatus = 'unverified_at_scan'

#: WHY a fact is ``unverified_at_scan``, and the two are not the same fact about the contract.
#: ``not_derivable`` is expected and permanent — a behavioural claim cannot be decided by
#: reading a corpus, and stamping it is the honest outcome. ``unresolvable`` is a DEFECT IN THE
#: CONTRACT ITSELF: the author declared a corpus-derivable check that cannot execute (no terms,
#: no expected_count, or a scope that addresses no corpus text). Collapsing the two is how a
#: scan whose every check was vacuous renders as an all-clear: a scope glob that selects nothing
#: leaves every fact "unverified" under a PROCEED headline.
UNVERIFIED_NOT_DERIVABLE = 'not_derivable'
UNVERIFIED_UNRESOLVABLE = 'unresolvable'
#: ...and the third, which is neither a defect nor a permanent impossibility: the author declared
#: the fact and no executable check for it, so it is ASSERTED and this scan has no static route to
#: decide it. Distinct from ``not_derivable`` because a behavioural claim can never be decided by
#: reading anything, while this one could be — by a check nobody wrote, or by exercising the code.
#: It is what makes a PROCEED provisional rather than clean.
UNVERIFIED_NO_STATIC_ROUTE = 'no_static_route'


#: Severity ranking, highest first — the HOUSE vocabulary, read off the schema Literal rather
#: than restated. Used to order probe budgets and report items; it is the fact AUTHOR's
#: declaration, never something inferred from the text. Deriving it means a severity added to the
#: schema is immediately rankable here instead of silently sorting last.
SEVERITY_ORDER: tuple[str, ...] = get_args(Severity)

#: Prefix for an id this loader DERIVES from a fact's own content when the author declared none.
#: Content-addressed on purpose: a positional id (``fact_00``, ``fact_01``) would silently
#: repoint every ``grounded_on`` stamp in the suite at a different claim when the facts document
#: is reordered — a provenance link that survives a reorder is the only kind worth stamping. The
#: ``h`` keeps a derived id from ever colliding with the positional spelling, which is how
#: :func:`fact_id_is_positional` can still recognise one.
DERIVED_FACT_ID_PREFIX = 'fact_h'

#: The positional spelling (``fact_00``), recognised so a stamping consumer can refuse it.
_POSITIONAL_FACT_ID = re.compile(r'fact_\d+')

#: Origins that the corpus OUTRANKS on a corpus-derivable fact. A fact declared as read off the
#: corpus that the corpus contradicts is a different failure (a mis-transcription), and says so.
_OUTRANKED_ORIGINS = frozenset({'prd', 'brief', 'eval_intent', 'interview', 'assumption', 'unstated'})

#: How many contradicting units one verdict quotes. Bounded because evidence payloads can carry
#: real customer records, and because ten quotes of the same mistake are not ten findings.
_MAX_EVIDENCE = 3
#: How much of a record one excerpt may carry — the house cap, imported rather than re-chosen.
#: Two independent limits on how much of a customer record a report may quote is one limit that
#: does not move when the other does. (The WINDOW is this module's own: an excerpt centred on the
#: match, not a head cut, which is why ``suite_findings.excerpt`` itself is not reused.)
_EXCERPT_CHARS = SUITE_EXCERPT_CHARS

#: Codepoint ranges per writing system, checked in order. Deliberately a table and not
#: ``unicodedata`` (which exposes no script property) nor a third-party dependency: the whole
#: package is stdlib + pyyaml, and the question asked here — "could a literal written in THIS
#: alphabet ever occur in THAT text" — needs alphabet granularity, not per-language precision.
#: Digits, punctuation and whitespace belong to no script and are ignored on both sides, which
#: is what keeps a URL or a product code from making an English term look at home in Hebrew prose.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ('latin', ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF))),
    ('greek', ((0x370, 0x3FF), (0x1F00, 0x1FFF))),
    ('cyrillic', ((0x400, 0x52F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ('hebrew', ((0x590, 0x5FF), (0xFB1D, 0xFB4F))),
    ('arabic', ((0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ('devanagari', ((0x900, 0x97F),)),
    ('thai', ((0xE00, 0xE7F),)),
    # The four SPACELESS abugidas of mainland South-East Asia and the Himalaya. Present because
    # the boundaryless-script set below names all four, and a script that writes no word gaps
    # and that this table cannot classify makes every whitespace-based measurement read a whole
    # reply as one token, silently — the exact blindness that set exists to catch. A name in that
    # set that this table cannot produce would be unreachable and the set would claim coverage it
    # does not have.
    ('lao', ((0xE80, 0xEFF),)),
    ('tibetan', ((0xF00, 0xFFF),)),
    ('myanmar', ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F))),
    ('khmer', ((0x1780, 0x17FF), (0x19E0, 0x19FF))),
    ('han', ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ('kana', ((0x3040, 0x30FF), (0x31F0, 0x31FF))),
    ('hangul', ((0x1100, 0x11FF), (0xAC00, 0xD7AF))),
)


#: The scripts that write WITHOUT a space between words: Japanese kana, Han, Thai, Lao, Myanmar,
#: Khmer, Tibetan. Hangul is deliberately absent — Korean uses a syllabic block script but DOES
#: space its words, so treating a Korean line break as no boundary would glue two words.
#:
#: ONE spelling for this package and for the backend's measurement primitives that re-export it.
#: Every name here must be one :data:`_SCRIPT_RANGES` can produce, and
#: :func:`boundaryless_character_class` derives the regex from the table rather than restating it.
BOUNDARYLESS_SCRIPTS: frozenset[str] = frozenset({'han', 'kana', 'thai', 'lao', 'khmer', 'myanmar', 'tibetan'})


def boundaryless_character_class() -> str:
    """A regex character-class BODY covering every codepoint of :data:`BOUNDARYLESS_SCRIPTS`.

    Derived from :data:`_SCRIPT_RANGES` so the two can never disagree: a range added to the table
    for one of these scripts joins the class in the same edit. The endpoints are all far above
    ASCII, so none needs escaping inside ``[...]``.
    """
    return ''.join(f'{chr(low)}-{chr(high)}' for name, ranges in _SCRIPT_RANGES if name in BOUNDARYLESS_SCRIPTS for low, high in ranges)


#: Every writing-system name :func:`_script_of` can actually return. DERIVED from the table, so a
#: closed set built over it — "the scripts that are not latin", "the ones that write no word
#: gaps" — grows with the table in the same edit instead of quietly shrinking to whatever was
#: true when it was typed.
SCRIPT_NAMES: frozenset[str] = frozenset(name for name, _ in _SCRIPT_RANGES)


#: What :func:`declared_language_scripts` answers for a declared language it cannot place. NOT a
#: member of :data:`SCRIPT_NAMES` — no character is ever classified as it — so a consumer that
#: compares it against a census can never mistake it for a script the sources are written in.
SCRIPT_UNKNOWN = 'unknown'

#: ISO 15924 script code → the :data:`SCRIPT_NAMES` its text is classified as. A script is a SET
#: for exactly the composite codes: Japanese (``Jpan``) is Han plus both kana. A code absent here
#: names a script :data:`_SCRIPT_RANGES` cannot classify, and resolves to :data:`SCRIPT_UNKNOWN`.
_ISO15924_SCRIPTS: dict[str, frozenset[str]] = {
    'Latn': frozenset({'latin'}),
    'Grek': frozenset({'greek'}),
    'Cyrl': frozenset({'cyrillic'}),
    'Hebr': frozenset({'hebrew'}),
    'Arab': frozenset({'arabic'}),
    'Deva': frozenset({'devanagari'}),
    'Thai': frozenset({'thai'}),
    'Laoo': frozenset({'lao'}),
    'Tibt': frozenset({'tibetan'}),
    'Mymr': frozenset({'myanmar'}),
    'Khmr': frozenset({'khmer'}),
    'Hani': frozenset({'han'}),
    'Hans': frozenset({'han'}),
    'Hant': frozenset({'han'}),
    'Hira': frozenset({'kana'}),
    'Kana': frozenset({'kana'}),
    'Hrkt': frozenset({'kana'}),
    'Jpan': frozenset({'han', 'kana'}),
    'Hang': frozenset({'hangul'}),
    'Kore': frozenset({'hangul', 'han'}),
}

#: A declared LANGUAGE → the ISO 15924 script it is written in: ``(ISO 639-1, script, other
#: spellings)``, the other spellings being its ISO 639-2/T and 639-2/B codes, deprecated 639-1
#: codes (``iw``, ``in``, ``ji``), its English name and its endonym — all casefolded.
#:
#: A CLOSED table, and why one exists here: the backend derives this answer from CLDR (``babel``), but this package is vendored into the client SDK
#: stdlib-only, so it cannot. Every row is CLDR's answer copied, not an opinion, and a parity
#: check re-derives each one from CLDR so the copy cannot drift. The ONE row that departs from
#: CLDR's likely script is Korean: CLDR names the composite ``Kore`` (Hangul + Han), but modern Korean prose carries almost no Han, and declaring
#: Han for a Korean agent would refuse every detector over a Korean corpus for lacking it.
#:
#: A language absent here is :data:`SCRIPT_UNKNOWN` — never Latin. A declaration such as ``he``,
#: ``Hebrew`` or ``ja`` is written in Latin characters whatever script the language itself uses,
#: so its own characters say nothing about the script.
_LANGUAGE_SCRIPT_ROWS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('en', 'Latn', ('eng', 'english')),
    ('es', 'Latn', ('spa', 'spanish', 'español')),
    ('fr', 'Latn', ('fra', 'fre', 'french', 'français')),
    ('de', 'Latn', ('deu', 'ger', 'german', 'deutsch')),
    ('it', 'Latn', ('ita', 'italian', 'italiano')),
    ('pt', 'Latn', ('por', 'portuguese', 'português')),
    ('nl', 'Latn', ('nld', 'dut', 'dutch', 'nederlands')),
    ('sv', 'Latn', ('swe', 'swedish', 'svenska')),
    ('no', 'Latn', ('nor', 'norwegian', 'norsk')),
    ('nb', 'Latn', ('nob', 'norwegian bokmål', 'norsk bokmål')),
    ('nn', 'Latn', ('nno', 'norwegian nynorsk', 'norsk nynorsk')),
    ('da', 'Latn', ('dan', 'danish', 'dansk')),
    ('fi', 'Latn', ('fin', 'finnish', 'suomi')),
    ('pl', 'Latn', ('pol', 'polish', 'polski')),
    ('cs', 'Latn', ('ces', 'cze', 'czech', 'čeština')),
    ('sk', 'Latn', ('slk', 'slo', 'slovak', 'slovenčina')),
    ('ro', 'Latn', ('ron', 'rum', 'romanian', 'română')),
    ('hu', 'Latn', ('hun', 'hungarian', 'magyar')),
    ('tr', 'Latn', ('tur', 'turkish', 'türkçe')),
    ('id', 'Latn', ('ind', 'in', 'indonesian')),
    ('ms', 'Latn', ('msa', 'may', 'malay', 'melayu')),
    ('vi', 'Latn', ('vie', 'vietnamese', 'tiếng việt')),
    ('fil', 'Latn', ('tl', 'tgl', 'filipino', 'tagalog')),
    ('ca', 'Latn', ('cat', 'catalan', 'català')),
    ('hr', 'Latn', ('hrv', 'croatian', 'hrvatski')),
    ('sl', 'Latn', ('slv', 'slovenian', 'slovenščina')),
    ('et', 'Latn', ('est', 'estonian', 'eesti')),
    ('lv', 'Latn', ('lav', 'latvian', 'latviešu')),
    ('lt', 'Latn', ('lit', 'lithuanian', 'lietuvių')),
    ('sq', 'Latn', ('sqi', 'alb', 'albanian', 'shqip')),
    ('sw', 'Latn', ('swa', 'swahili', 'kiswahili')),
    ('af', 'Latn', ('afr', 'afrikaans')),
    ('ga', 'Latn', ('gle', 'irish', 'gaeilge')),
    ('eu', 'Latn', ('eus', 'baq', 'basque', 'euskara')),
    ('gl', 'Latn', ('glg', 'galician', 'galego')),
    ('is', 'Latn', ('isl', 'ice', 'icelandic', 'íslenska')),
    ('cy', 'Latn', ('cym', 'wel', 'welsh', 'cymraeg')),
    ('mt', 'Latn', ('mlt', 'maltese', 'malti')),
    ('he', 'Hebr', ('heb', 'iw', 'hebrew', 'עברית')),
    ('yi', 'Hebr', ('yid', 'ji', 'yiddish', 'ייִדיש')),
    ('ar', 'Arab', ('ara', 'arabic', 'العربية')),
    ('fa', 'Arab', ('fas', 'per', 'persian', 'فارسی')),
    ('ur', 'Arab', ('urd', 'urdu', 'اردو')),
    ('ps', 'Arab', ('pus', 'pashto', 'پښتو')),
    ('ru', 'Cyrl', ('rus', 'russian', 'русский')),
    ('uk', 'Cyrl', ('ukr', 'ukrainian', 'українська')),
    ('be', 'Cyrl', ('bel', 'belarusian', 'беларуская')),
    ('bg', 'Cyrl', ('bul', 'bulgarian', 'български')),
    ('sr', 'Cyrl', ('srp', 'serbian', 'српски')),
    ('mk', 'Cyrl', ('mkd', 'mac', 'macedonian', 'македонски')),
    ('kk', 'Cyrl', ('kaz', 'kazakh', 'қазақ тілі')),
    ('ky', 'Cyrl', ('kir', 'kyrgyz', 'кыргызча')),
    ('mn', 'Cyrl', ('mon', 'mongolian', 'монгол')),
    ('tg', 'Cyrl', ('tgk', 'tajik', 'тоҷикӣ')),
    ('el', 'Grek', ('ell', 'gre', 'greek', 'ελληνικά')),
    ('hi', 'Deva', ('hin', 'hindi', 'हिन्दी')),
    ('mr', 'Deva', ('mar', 'marathi', 'मराठी')),
    ('ne', 'Deva', ('nep', 'nepali', 'नेपाली')),
    ('th', 'Thai', ('tha', 'thai', 'ไทย')),
    ('lo', 'Laoo', ('lao', 'ລາວ')),
    ('my', 'Mymr', ('mya', 'bur', 'burmese', 'မြန်မာ')),
    ('km', 'Khmr', ('khm', 'khmer', 'ខ្មែរ')),
    ('bo', 'Tibt', ('bod', 'tib', 'tibetan')),
    ('zh', 'Hans', ('zho', 'chi', 'chinese', '中文')),
    ('ja', 'Jpan', ('jpn', 'japanese', '日本語')),
    ('ko', 'Hang', ('kor', 'korean', '한국어')),
)

#: Every spelling in :data:`_LANGUAGE_SCRIPT_ROWS` → its ISO 15924 code, derived once.
_LANGUAGE_SCRIPT_INDEX: dict[str, str] = {spelling: script for code, script, others in _LANGUAGE_SCRIPT_ROWS for spelling in (code, *others)}


def declared_language_scripts(language: str | None) -> frozenset[str]:
    """The :data:`SCRIPT_NAMES` a DECLARED language is written in.

    Accepts what a declaration actually carries: an ISO 639-1/-2 code (``he``, ``heb``), an
    English name (``Hebrew``), an endonym (``עברית``), or a BCP-47 tag with a region and/or script
    subtag (``he-IL``, ``pt_BR``, ``zh-Hant``, ``sr-Latn``) — an explicit script subtag wins over the
    language's usual script, because that is what it declares.

    Empty for a blank declaration (nothing was declared). ``{SCRIPT_UNKNOWN}`` for one this table
    cannot place, or whose script :data:`_SCRIPT_RANGES` cannot classify — never a guess, and in
    particular never Latin: the caller reports an unknown declaration as unmeasured instead of
    measuring the wrong script.
    """
    text = ' '.join((language or '').split()).casefold()
    if not text:
        return frozenset()
    code = _LANGUAGE_SCRIPT_INDEX.get(text)
    if code is None:
        base, *subtags = text.replace('_', '-').split('-')
        script_subtag = next((tag.title() for tag in subtags if len(tag) == 4 and tag.isalpha()), None)
        code = script_subtag or _LANGUAGE_SCRIPT_INDEX.get(base)
    return _ISO15924_SCRIPTS.get(code or '', frozenset({SCRIPT_UNKNOWN}))


#: Share of a text's SCRIPTED characters a writing system must carry to count as one the text is
#: actually written in. Not "present at all": a Hebrew knowledge base routinely carries Latin
#: product names, URLs and code identifiers, and a presence test would call every English term
#: at home there — which is precisely the vacuous verification this guard exists to catch.
SCRIPT_DOMINANCE_SHARE = 0.20


#: The ASCII fast path for :func:`fold_key`. ``[\W_]+`` and the category rule below agree
#: EXACTLY on ASCII, which carries no combining marks and no case-folding expansions — so this is
#: a speed shortcut and never a second rule.
_ASCII_NON_KEY_RE = re.compile(r'[\W_]+')

#: The Unicode general-category MAJORS a key character belongs to: Letter, Number, Mark.
#:
#: ``M`` is the half ``\w`` gets wrong and the reason this is a category test rather than a regex
#: class. Python's ``\w`` under ``str`` patterns is Unicode-aware about LETTERS but excludes the
#: combining marks (``Mn``/``Mc``/``Me``) that carry the vowels in an abugida and the diacritics
#: in a pointed abjad — so ``[\W_]+`` would treat every Thai vowel sign, every Devanagari matra,
#: every Arabic harakat and every Hebrew niqqud as a WORD SEPARATOR. A key built on it would split
#: mid-word, and a dedup keyed on it would merge unrelated fragments and split identical terms.
#:
#: A mark is BY DEFINITION attached to the base character before it; that is what makes this
#: derivable rather than a table: no script list, no codepoint ranges, just the property
#: Unicode already publishes for every character.
_KEY_CATEGORY_MAJORS = frozenset('LNM')


def fold_key(text: str) -> str:
    r"""The universal normalization key: casefolded, with runs of non-key characters collapsed.

    Use it instead of the ``re.sub(r'[^a-z0-9]+', ' ', s.lower())`` form wherever a KEY is being
    built. That ASCII form maps EVERY non-Latin string to the empty string, so a dedup keyed on it
    silently merges unrelated Hebrew, Japanese and Thai values into one bucket. Callers that dedup
    on this key must still refuse to merge on an EMPTY key — an unkeyable value is unknown, not identical.

    A key character is a LETTER, a NUMBER **or a MARK** (:data:`_KEY_CATEGORY_MAJORS`) — read off
    ``unicodedata.category``, not off ``\w``. ``\w`` excludes combining marks, so ``[\W_]+``
    would not erase non-Latin strings but would CUT them, at every vowel sign in an abugida and every diacritic in a pointed
    abjad. ``การคืนเงิน`` ("the refund") would key as three words; a policy key built from it could
    never match itself.

    Everything else — punctuation, symbols, whitespace, ``_`` — is a separator, and runs of them
    collapse to a single space.
    """
    folded = (text or '').casefold()
    if folded.isascii():
        # Identical result, one C-level pass. ASCII has no marks, so the two rules cannot diverge.
        return _ASCII_NON_KEY_RE.sub(' ', folded).strip()
    out: list[str] = []
    separated = True  # suppresses a leading space without a second pass
    for char in folded:
        if unicodedata.category(char)[0] in _KEY_CATEGORY_MAJORS:
            out.append(char)
            separated = False
        elif not separated:
            out.append(' ')
            separated = True
    return ''.join(out).rstrip()


#: The closed-class detector LADDER's rungs, in the order it descends them. Owned here rather
#: than in :mod:`plurai.scan_contract.closed_classes`, which implements them, because this module
#: has to VALIDATE a customer-declared ``candidate.rung`` and ``closed_classes`` imports this one
#: (for the script table and ``fold_key``), so the dependency can only run this way.
RUNG_SHAPE = 'shape'
RUNG_INDUCED = 'induced'
RUNG_NEAR_MISS = 'near_miss'
#: ...and the fourth, which is not a rung the ladder can be asked for: it is what a class gets when
#: every rung refused. A customer cannot declare it as a candidate, so it is outside
#: :data:`DETECTOR_RUNGS`.
RUNG_NONE = 'none'
DETECTOR_RUNGS: tuple[str, ...] = (RUNG_SHAPE, RUNG_INDUCED, RUNG_NEAR_MISS)


#: The PREDICATE a ``data_profile`` fact carries when it declares a closed set. The only predicate
#: a ``detector:`` block is legal on: everything else is a claim about presence or shape, and a
#: detector is a claim about EXHAUSTIVENESS. ``lint_prd.py`` refuses the block anywhere else, and
#: this constant is the one spelling the two ends share.
PREDICATE_CLOSED_SET = 'closed_set'
#: The predicate of a ``rule`` fact: the agent's own instruction, quoted. Its subject is its own
#: ``path:LINE``, not an identifier, and the quote is the text the agent is given.
PREDICATE_INSTRUCTS = 'instructs'


@dataclass(frozen=True)
class DetectorCandidate:
    """What the SCAN lifted from the customer's own code as a candidate detector, with provenance.

    A schema ``Literal[...]``, a validation regex, a link constant, a constant collection of
    codes — each is a detector the customer already wrote. It is a PROPOSAL, never
    an active detector: the six gates still run over it, and a candidate lifted from an enum
    passes them easily while one induced from three product codes will not, and should not.
    """

    rung: str
    pattern: str = ''
    source: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {'rung': self.rung, 'pattern': self.pattern, 'source': self.source}


@dataclass(frozen=True)
class DetectorRequirement:
    """Whether a closed-set fact MUST have a deterministic detector, and what may control it.

    The declaration that makes a missing check impossible to ship silently: a closed class is
    never a licence with nothing behind it unless someone says so. "Closed but judge-only" stays
    available; it just has to be said out loud, by a human, in the customer's own repository.
    """

    required: bool
    #: Literals the author asserts are NOT members. At least one is mandatory when ``required``:
    #: specificity proves the detector fires on nothing wrong, these prove it fires on something
    #: right, and a requirement with nothing to control against is a rubber stamp.
    negatives: tuple[str, ...] = ()
    candidate: DetectorCandidate | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {'required': self.required, 'negatives': list(self.negatives)}
        if self.candidate is not None:
            out['candidate'] = self.candidate.as_dict()
        return out


@dataclass(frozen=True)
class ScanFact:
    """One typed scan conclusion, carrying its own machine-checkable payload.

    The ``claim`` is the human sentence; the typed payload is what makes the claim EXECUTABLE.
    A fact with a derivable ``kind`` and no payload cannot be checked, and is reported as such
    rather than counted as verified — a checker that passes what it cannot read is the
    coverage theatre this package exists to remove.
    """

    fact_id: str
    kind: str
    claim: str
    #: Which document asserted this. Decides precedence when the corpus disagrees.
    origin: str = 'unstated'
    #: WHERE the author read or measured it, verbatim — ``tools/catalog.py:214``,
    #: ``artifacts/data_profile.md``, ``corpus:<doc>``. ``origin`` is the arbitration class this
    #: collapses to; the citation itself is what :mod:`plurai.scan_contract.citations` resolves,
    #: and collapsing it away is why a PRD could cite a line eleven off and read clean.
    source: str = ''
    #: The machine-derived name the fact is ABOUT (``<tool>``, ``<entity>.<field>``). Carried so a
    #: ``path:LINE`` citation can be checked for supporting the claim rather than merely existing.
    subject: str = ''
    #: Literals the check matches on, case-insensitively, over whitespace-normalized text.
    terms: tuple[str, ...] = ()
    #: ``any`` — one term is enough; ``all`` — every term must occur in the SAME unit.
    match: str = 'any'
    #: Restrict matching to flattened structured field(s): an exact dotted path, or an fnmatch
    #: glob addressing one column across every record (``assets.*.link_url``). ``None`` = whole unit.
    field_name: str | None = None
    #: fnmatch globs over unit ids. Empty = the whole corpus.
    scope: tuple[str, ...] = ()
    #: Required for ``count``: how many units in scope must match.
    expected_count: int | None = None
    severity: str = 'medium'
    #: Optional author-written probe question, used verbatim when this fact generates a probe.
    probe_ask: str | None = None
    #: The predicate a ``data_profile`` fact declares (``closed_set``, ``absent_from_source``, …).
    #: Carried so the detector gate can find the closed-set facts without re-deriving the
    #: predicate from the id the projection happens to have built out of it.
    predicate: str = ''
    #: Whether this closed set must have a deterministic detector, and what controls it. ``None``
    #: is a THIRD state, distinct from ``required: false``: nobody said, which the report files as
    #: advisory rather than as a choice the customer made.
    detector: DetectorRequirement | None = None

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.index(self.severity) if self.severity in SEVERITY_ORDER else len(SEVERITY_ORDER)

    def check_payload(self) -> dict[str, Any]:
        """The EXECUTABLE half of this fact — what a second checker needs to re-decide it.

        Published on every verdict so a downstream consumer can re-run the same check against a
        DIFFERENT corpus (the server-side registry the suite is actually grounded
        on, as opposed to the client dirs the scan read). Without it the artifact carries only
        the human sentence and the verdict, and a reconciler can compare nothing.
        """
        return {
            'terms': list(self.terms),
            'match': self.match,
            'field': self.field_name,
            'scope': list(self.scope),
            'expected_count': self.expected_count,
        }


@dataclass(frozen=True)
class FactEvidence:
    """One corpus unit that decided a verdict, with a short quote around the match."""

    unit_id: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return {'unit_id': self.unit_id, 'excerpt': self.excerpt}


@dataclass(frozen=True)
class FactVerdict:
    """What the checker established about ONE fact."""

    fact: ScanFact
    status: FactStatus
    reason: str
    evidence: tuple[FactEvidence, ...] = ()
    #: Set only when ``status`` is ``unverified_at_scan``: which of the two reasons applies.
    unverified_kind: str | None = None
    #: How many ``(label, text)`` scopes the fact's scope + field selected. Zero is reported,
    #: not assumed benign: an absence claim over an empty scope is vacuously true and proves nothing.
    scopes_addressed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            'fact_id': self.fact.fact_id,
            'kind': self.fact.kind,
            'origin': self.fact.origin,
            # Published so ``load_scan_facts`` round-trips an artifact back into the facts that
            # produced it — the citation audit re-runs off this list, and a verdict that drops
            # the citation leaves it with nothing to resolve.
            'source': self.fact.source,
            'subject': self.fact.subject,
            'severity': self.fact.severity,
            'claim': self.fact.claim,
            'status': self.status,
            'unverified_kind': self.unverified_kind,
            'reason': self.reason,
            'scopes_addressed': self.scopes_addressed,
            'evidence': [e.as_dict() for e in self.evidence],
            # Nested, not flattened, so the artifact says which keys are the CHECK rather than
            # leaving a reader to guess which of a verdict's fields are inputs. ``load_scan_facts``
            # reads either spelling.
            'check': self.fact.check_payload(),
        }


@dataclass
class FactVerification:
    """Every verdict plus the coverage statement that keeps them honest."""

    verdicts: list[FactVerdict] = field(default_factory=list)

    def of_status(self, status: FactStatus) -> list[FactVerdict]:
        return [v for v in self.verdicts if v.status == status]

    def unverified_of_kind(self, kind: str) -> list[FactVerdict]:
        return [v for v in self.verdicts if v.status == UNVERIFIED_AT_SCAN and v.unverified_kind == kind]

    @property
    def flagged(self) -> list[FactVerdict]:
        """Every fact the check has something to say about: a false fact, or a declared check that could not
        execute. Only the first HOLDS anything (the report's ``scan.fact_contradicted``); both are findings."""
        return [v for v in self.verdicts if v.status == CONTRADICTED or v.unverified_kind == UNVERIFIED_UNRESOLVABLE]

    @property
    def contradicted(self) -> list[FactVerdict]:
        return self.of_status(CONTRADICTED)

    @property
    def counts(self) -> dict[str, int]:
        return {status: len(self.of_status(status)) for status in (VERIFIED, CONTRADICTED, UNVERIFIED_AT_SCAN)}  # type: ignore[arg-type]

    def coverage_note(self) -> str:
        """What this verification does and does NOT prove. Rendered into every report.

        Stated as a sentence rather than left to the counts, because the counts alone invite
        exactly the misreading this check must not license: that a clean run says anything
        about the claims it could not decide.
        """
        total = len(self.verdicts)
        not_derivable = len(self.unverified_of_kind(UNVERIFIED_NOT_DERIVABLE))
        unresolvable = len(self.unverified_of_kind(UNVERIFIED_UNRESOLVABLE))
        asserted = len(self.unverified_of_kind(UNVERIFIED_NO_STATIC_ROUTE))
        note = f'{total - not_derivable - unresolvable - asserted} of {total} declared fact(s) were DECIDED against the corpus'
        if not_derivable:
            note += f'; {not_derivable} are behavioural and not decidable by reading a corpus — this check proves NOTHING about them'
        if asserted:
            note += f'; {asserted} are ASSERTED with no executable check — unverified, and generation will build on them anyway'
        if unresolvable:
            note += (
                f'; {unresolvable} declare a corpus-derivable check that COULD NOT EXECUTE (missing payload, or a scope '
                'addressing no corpus text) — the contract is misconfigured, not clean'
            )
        return note + '.'

    def as_dict(self) -> dict[str, Any]:
        return {'counts': self.counts, 'coverage_note': self.coverage_note(), 'verdicts': [v.as_dict() for v in self.verdicts]}


def _listed(value: Any) -> list[Any]:
    """``value`` as the list :func:`derived_fact_id` spells, for any shape a malformed row carries."""
    if isinstance(value, (list, tuple, str)):
        return list(value)
    return [] if not value else [value]


def derived_fact_id(row: Mapping[str, Any]) -> str:
    """A stable id for a fact whose author declared none, derived from what the fact SAYS.

    Keyed on the claim plus the executable check, in a canonical spelling, so the same fact
    keeps the same id no matter where it sits in the document. An id derived from the row's
    index would make every ``grounded_on`` stamp in a suite point at whatever fact later took
    that slot, which is a provenance record that lies after the first edit.
    """
    canonical = json.dumps(
        {
            'kind': str(row.get('kind') or KIND_BEHAVIOURAL),
            'claim': ' '.join(str(row.get('claim') or '').split()),
            'terms': [str(t) for t in _listed(row.get('terms'))],
            'match': str(row.get('match') or 'any'),
            'field': str(row['field']) if row.get('field') else None,
            'scope': [str(s) for s in _listed(row.get('scope'))],
            'expected_count': row.get('expected_count'),
        },
        sort_keys=True,
        ensure_ascii=False,
        # A malformed row still needs a NAME, so its drop can be reported: whatever YAML produced is
        # spelled, never raised on (see ``load_scan_facts``).
        default=str,
    )
    return DERIVED_FACT_ID_PREFIX + hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:10]


def fact_id_is_positional(fact_id: str) -> bool:
    """Whether ``fact_id`` is the index-derived spelling (``fact_00``).

    Such an id names a SLOT, not a claim: reorder the facts document and it addresses a
    different fact while every stamp that recorded it still reads as valid. Anything built on
    fact provenance has to refuse it rather than record a link it cannot keep — see
    :attr:`plurai.scan_contract.contract.FactDependents.unstampable`.

    Matches on SHAPE, so an author-declared all-digit id (``fact_2024``) is treated as
    positional too. That is the safe direction: the consequence is a refused provenance stamp
    with a message naming the id, where the other direction is a stamp that silently repoints.
    """
    return bool(_POSITIONAL_FACT_ID.fullmatch(fact_id))


@dataclass(frozen=True)
class DroppedFact:
    """A declared fact whose payload could not be read, so it was left out of the scan — never silently."""

    fact_id: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {'fact_id': self.fact_id, 'reason': self.reason}


def load_scan_facts(payload: Any) -> tuple[list[ScanFact], list[DroppedFact]]:
    """Parse a typed-facts document (``{'facts': [...]}`` or a bare list) into :class:`ScanFact`.

    Returns ``(facts, dropped)``. ONE bad fact drops THAT fact, with its reason, and the rest are
    read: a malformed ``match``, ``terms``, ``expected_count``, ``check:`` or ``detector:`` block is
    a defect in one declaration, and refusing the whole document over it decided nothing about the
    others. A malformed check is DROPPED, never read with a guess: an unknown ``check`` key (a typo
    such as ``term:`` leaves the check matching more than meant), several terms with no explicit
    ``match``, or an ``expected_count`` that is not an integer would each be decided over a wider
    or different predicate than the author wrote, and a false VERIFIED is the silent result. Only
    the DOCUMENT's shape still raises ``ValueError`` — not a list (or a mapping with a ``facts``
    list), or an entry that is not a mapping — because nothing below it can be read at all.

    An entry with an unknown ``kind`` is admitted rather than dropped: the author declared a fact,
    and silently discarding it would shrink the report to look cleaner than the input was. It
    surfaces as ``unverified_at_scan`` with the unknown kind named — ``unresolvable`` when it carries
    ``terms`` or a ``scope`` (a check meant to run under a misspelled kind, named in the headline),
    ``not_derivable`` otherwise.
    """
    rows = payload.get('facts') if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("typed-facts document must be a list of facts, or a mapping with a 'facts' list.")
    facts: list[ScanFact] = []
    dropped: list[DroppedFact] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f'typed-facts entry {index} is not a mapping.')
        # A published VERDICT nests its executable half under ``check`` (see
        # ``ScanFact.check_payload``); a hand-authored facts document states the same keys at the
        # top level. Both are read, so an artifact round-trips back into the facts that produced it.
        entry: Mapping[str, Any] = {**row, **nested} if isinstance(nested := row.get('check'), Mapping) else row
        # Derived BEFORE the payload is validated, so every error below can name the fact it is
        # about. The derivation reads the raw row defensively and never raises.
        fact_id = str(entry.get('id') or entry.get('fact_id') or derived_fact_id(entry))
        try:
            facts.append(_load_fact(entry, fact_id, check=row.get('check')))
        except (ValueError, TypeError) as exc:
            dropped.append(DroppedFact(fact_id=fact_id, reason=str(exc)))
    return facts, dropped


#: Every key a ``check:`` block may carry — the executable half (:meth:`ScanFact.check_payload`) plus
#: the per-fact settings an authoring ``check:`` declares beside it. Closed: an unknown key is a typo
#: whose intent this reader cannot know, so the fact is dropped rather than decided without it.
CHECK_KEYS: frozenset[str] = frozenset({'kind', 'terms', 'match', 'scope', 'field', 'expected_count', 'severity', 'probe_ask'})


def _load_fact(entry: Mapping[str, Any], fact_id: str, *, check: Any) -> ScanFact:
    """One fact's payload, read. Raises ``ValueError`` / ``TypeError`` naming ``fact_id`` on anything malformed.

    ``check`` is the row's raw ``check:`` value (``None`` when the keys sit at the top level), validated
    as a closed block before it is read.
    """
    if check is not None and not isinstance(check, Mapping):
        raise ValueError(f'fact {fact_id!r}: check must be a mapping of {sorted(CHECK_KEYS)}, got {type(check).__name__}.')
    if isinstance(check, Mapping) and (unknown := sorted(str(k) for k in check if k not in CHECK_KEYS)):
        raise ValueError(
            f'fact {fact_id!r}: check carries unknown key(s) {unknown} (a check reads only {sorted(CHECK_KEYS)}); '
            'deciding it without them could match a wider scope than the author meant.'
        )
    terms = entry.get('terms') or []
    if not isinstance(terms, (list, tuple)):
        raise ValueError(f'fact {fact_id!r}: terms must be a list of strings, got {type(terms).__name__}.')
    if blank := [i for i, t in enumerate(terms) if not str(t).strip()]:
        raise ValueError(f'fact {fact_id!r}: terms {blank} are blank. Dropping them silently would weaken a match=all check to fewer terms than declared.')
    if len(terms) > 1 and entry.get('match') is None:
        # Never defaulted over several terms: `any` would verify an "all of these exist" claim on one
        # of them, under the author's own sentence.
        raise ValueError(f"fact {fact_id!r}: {len(terms)} terms need an explicit match — 'any' (one term is enough) or 'all' (every term).")
    match = str(entry.get('match') or 'any')
    if match not in ('any', 'all'):
        # Never defaulted. Anything not exactly 'all' would silently mean 'any' — the weaker
        # predicate — so a fact meaning to require every term would verify on one of them.
        raise ValueError(f"fact {fact_id!r}: match must be 'any' or 'all', got {match!r}.")
    scope = entry.get('scope') or []
    if not isinstance(scope, (list, tuple)):
        raise ValueError(f'fact {fact_id!r}: scope must be a list of unit-id globs, got {type(scope).__name__}.')
    expected = entry.get('expected_count')
    # NOT coerced: ``int(True)`` is 1 and ``int('3')`` reads a string, so either would count against
    # a number the author never wrote as one.
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
        raise ValueError(f'fact {fact_id!r}: expected_count must be an integer, got {expected!r}.')
    expected_count = expected
    return ScanFact(
        fact_id=fact_id,
        kind=str(entry.get('kind') or KIND_BEHAVIOURAL),
        claim=str(entry.get('claim') or ''),
        origin=str(entry.get('origin') or 'unstated'),
        source=str(entry.get('source') or ''),
        subject=str(entry.get('subject') or ''),
        terms=tuple(str(t) for t in terms),
        match=match,
        field_name=(str(entry['field']) if entry.get('field') else None),
        scope=tuple(str(s) for s in scope),
        expected_count=expected_count,
        severity=str(entry.get('severity') or 'medium'),
        probe_ask=(str(entry['probe_ask']) if entry.get('probe_ask') else None),
        predicate=str(entry.get('predicate') or ''),
        detector=_load_detector_requirement(entry.get('detector'), fact_id=fact_id, predicate=str(entry.get('predicate') or '')),
    )


def _load_detector_requirement(raw: Any, *, fact_id: str, predicate: str) -> DetectorRequirement | None:
    """Parse a fact's ``detector:`` block, or ``None`` when it declared none. Raises on malformed,
    which drops THIS fact (see :func:`load_scan_facts`) — never reads it with a guessed requirement.

    Fail closed, every time, because every silent default here is the wrong one:

    * ``required`` must be a REAL bool. ``required: "false"`` is truthy in Python, so a typo would
      turn "judge-only, by declaration" into a hard gate — and ``required: yes`` read as a string
      would do the reverse under a stricter reading.
    * ``negatives`` must be non-empty when ``required`` is true. A requirement with nothing to
      control the detector against cannot fail its own positive-control gate, which is what stops
      the whole mechanism from being a rubber stamp.
    * a ``detector:`` block is legal ONLY on ``predicate: closed_set``. On anything else it is an
      author reaching for a guarantee the predicate cannot carry, and admitting it would put a
      requirement on a fact nothing joins a class to.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f'fact {fact_id!r}: detector must be a mapping, got {type(raw).__name__}.')
    if predicate != PREDICATE_CLOSED_SET:
        raise ValueError(
            f'fact {fact_id!r}: a detector block is only meaningful on predicate {PREDICATE_CLOSED_SET!r}, not {predicate or "(none)"!r} — '
            'a detector is a claim about EXHAUSTIVENESS, and no other predicate makes one.'
        )
    required = raw.get('required')
    if not isinstance(required, bool):
        raise ValueError(
            f'fact {fact_id!r}: detector.required must be a real boolean, got {required!r}. A truthy string would turn '
            '"judge-only by declaration" into a hard gate, or the reverse, with nothing saying which.'
        )
    negatives = raw.get('negatives') or []
    if not isinstance(negatives, (list, tuple)) or any(not str(n).strip() for n in negatives):
        raise ValueError(f'fact {fact_id!r}: detector.negatives must be a list of non-blank literals the detector MUST catch.')
    if required and not negatives:
        raise ValueError(
            f'fact {fact_id!r}: detector.required is true, so detector.negatives needs at least one literal that is NOT a '
            'member. Without one the positive-control gate cannot run and the requirement is a rubber stamp.'
        )
    candidate = raw.get('candidate')
    if candidate is not None and not isinstance(candidate, Mapping):
        raise ValueError(f'fact {fact_id!r}: detector.candidate must be a mapping, got {type(candidate).__name__}.')
    parsed: DetectorCandidate | None = None
    if isinstance(candidate, Mapping):
        rung = str(candidate.get('rung') or '')
        if rung not in DETECTOR_RUNGS:
            raise ValueError(f'fact {fact_id!r}: detector.candidate.rung is {rung or "missing"!r}; the closed set is {", ".join(DETECTOR_RUNGS)}.')
        parsed = DetectorCandidate(rung=rung, pattern=str(candidate.get('pattern') or ''), source=str(candidate.get('source') or ''))
    return DetectorRequirement(required=required, negatives=tuple(str(n) for n in negatives), candidate=parsed)


@lru_cache(maxsize=4096)
def _script_of(char: str) -> str | None:
    """The writing system one character belongs to, or ``None`` for digits/punctuation/space.

    CACHED, because it is a per-character linear scan of the range table and every script question
    in the package runs it over whole corpora, asking the same handful of distinct characters again
    and again. The domain is one codepoint and the answer is a constant of the table, so the cache is
    exact; the bound is well past the size of any alphabet the table classifies.
    """
    code = ord(char)
    for name, ranges in _SCRIPT_RANGES:
        if any(low <= code <= high for low, high in ranges):
            return name
    return None


def scripts_present(text: str) -> set[str]:
    """Every writing system that occurs in ``text`` at all. Empty for digits/punctuation only."""
    return {script for ch in text if (script := _script_of(ch)) is not None}


#: An address token STOPS at whitespace or at the first character of a boundaryless script —
#: derived from :func:`boundaryless_character_class`, never restated. Japanese, Han, Thai, Khmer,
#: Lao, Myanmar and Tibetan write no spaces, so with ``\S*`` on both sides ONE inline link would
#: make the entire paragraph a single non-space run and the census below would count ZERO prose
#: characters — see :data:`_ADDRESS_TOKEN_RE`.
_ADDRESS_STOP = boundaryless_character_class()

#: Tokens whose letters are ADDRESSES rather than prose: URLs (``://``), emails and handles
#: (``@``), paths (``/`` or ``\\``), and dotted / underscored identifiers (``link_url``,
#: ``config.yaml``, ``example.com``). Structural — no TLD list, no vocabulary — because the
#: property that matters is that these letters are Latin BY PROTOCOL and say nothing about the
#: language the scope is written in.
#:
#: Excluding them keeps the census about prose. On a short Japanese help page, one
#: ``https://example.com/help/cancellation-policy`` adds enough Latin characters to clear
#: :data:`SCRIPT_DOMINANCE_SHARE`, so ``latin`` would count as a script the page "is written in",
#: :func:`script_mismatch` would find nothing wrong with an English term over it, and an English
#: ABSENCE fact would return VERIFIED. Catalog-shaped corpora (short fields beside a URL column)
#: sit in that regime by construction.
#:
#: BOUNDED at both ends by :data:`_ADDRESS_STOP` rather than by whitespace alone, because
#: Japanese is not spaced around a link. Written the way Japanese is actually written —
#: ``…ヘルプhttps://example.com/help/cancellation-policyをご覧ください。`` — a greedy ``\S*``
#: would swallow the paragraph on both sides of the link and the census would drop to 0 prose
#: characters: the vacuous-English-term guard would go dark on exactly its motivating case, AND
#: :func:`script_mismatch` would tell the operator a long page was "too short to establish a
#: script". Punctuation BETWEEN two boundaryless characters (``。``, ``、``, ``「``) is still
#: swallowed and deliberately so: it belongs to no script, so the census never counted it and
#: naming it here would be the fourth hand-rolled character table this package refuses to keep.
_ADDRESS_TOKEN_RE = re.compile(rf'[^\s{_ADDRESS_STOP}]*(?:://|[@\\/]|[^\W{_ADDRESS_STOP}][._][^\W{_ADDRESS_STOP}])[^\s{_ADDRESS_STOP}]*')

#: Scripted PROSE characters a scope must carry before a SHARE of them means anything — DERIVED
#: from :data:`SCRIPT_DOMINANCE_SHARE` rather than picked: at ``1 / SCRIPT_DOMINANCE_SHARE``
#: characters a SINGLE character already clears the threshold (1/5 = 0.20), so at or below that
#: count one stray letter "dominates" the scope. One more than that is the smallest census in
#: which the threshold is a threshold. Below it this function reports NO dominant script, and
#: :func:`script_mismatch` falls back to per-character reachability — abstaining out loud when the
#: terms share no script with the text — instead of letting the guard pass by default.
MIN_PROSE_SCRIPT_CHARS = int(1 / SCRIPT_DOMINANCE_SHARE) + 1


def prose_characters(text: str) -> str:
    """``text`` with its address-shaped tokens removed — see :data:`_ADDRESS_TOKEN_RE`."""
    return _ADDRESS_TOKEN_RE.sub(' ', text or '')


def scripted_prose_count(text: str) -> int:
    """How many scripted characters of ``text`` are PROSE — the census denominator."""
    return sum(1 for ch in prose_characters(text) if _script_of(ch) is not None)


def dominant_scripts(text: str) -> set[str]:
    """The writing systems ``text`` is actually WRITTEN in — each at least
    :data:`SCRIPT_DOMINANCE_SHARE` of its scripted PROSE characters.

    PROSE, not every character: a URL, an email, a path and a dotted identifier are Latin because
    of the protocols that spell them, and counting them let one link make ``latin`` a language a
    Japanese page is written in (see :data:`_ADDRESS_TOKEN_RE`).

    Returns the empty set for a text with no scripted prose at all (a pure catalog of ids) AND for
    one carrying fewer than :data:`MIN_PROSE_SCRIPT_CHARS` of them — too short for a share to mean
    anything. The caller must read the empty set as "cannot say", never as "no script matches";
    :func:`script_mismatch` is the one that turns it into a stated abstention.
    """
    counts: dict[str, int] = {}
    total = 0
    for ch in prose_characters(text):
        if (script := _script_of(ch)) is not None:
            counts[script] = counts.get(script, 0) + 1
            total += 1
    if total < MIN_PROSE_SCRIPT_CHARS:
        return set()
    return {script for script, n in counts.items() if n / total >= SCRIPT_DOMINANCE_SHARE}


def script_mismatch(terms: Sequence[str], match: str, text: str) -> str:
    """Why ``terms`` could never match ``text`` on alphabet grounds — or ``''`` when they could.

    Matching here is case-insensitive SUBSTRING, so a term written in an alphabet the addressed
    text does not use cannot match ANY of it. For an absence claim that produces a VERIFIED
    verdict — the silent direction, and the one that mints an adversarial probe whose ground
    truth is "the agent must decline". A corpus can be in a non-Latin alphabet; terms transcribed into
    English over it verify every absence in the contract and prove nothing about any of them.

    ``match`` decides how many terms have to be plausible. Under ``any`` one live term makes the
    check real. Under ``all`` every term must be able to occur, because one impossible term
    makes the conjunction unsatisfiable and the absence vacuous — the same asymmetry the matcher
    itself has.
    """
    text_scripts = dominant_scripts(text)
    if not text_scripts:
        # NO DOMINANT SCRIPT: a short scope (a price, an id, a code column) or a thoroughly mixed
        # one. A share cannot be measured, but REACHABILITY still can, character by character: a
        # term written in no script (digits and punctuation) can match any text, and a scripted
        # term can match only a text that carries its script at all. Same ``any``/``all`` rule as
        # the dominant branch below. Without this an absence over a numeric or id field
        # (``0.00`` over ``*.price``, ``SKU-9999`` over ``*.sku``) could never be established.
        present = scripts_present(text)
        reachable = [t for t in terms if not (term_scripts := scripts_present(t)) or term_scripts & present]
        if reachable and (match != 'all' or len(reachable) == len(terms)):
            return ''
        # ABSTAIN, LOUDLY. An empty census is "cannot say", and returning '' here would read it as
        # "no mismatch" — the permissive direction, under which an English absence over a scope
        # with no readable prose would come back VERIFIED. The caller (the absence arm of
        # ``verify_fact``) treats a non-empty reason as "this scope could not have matched", so the fact lands
        # UNRESOLVABLE rather than establishing a gap nobody looked for.
        #
        # BRANCHED on the census, because ``dominant_scripts`` returns the empty set for TWO
        # different scopes and only one of them is short. Printing "too short to establish a
        # script" over a long, thoroughly mixed page is a false sentence about the operator's own
        # corpus, and the one thing an abstention may not do is misdescribe what it abstained on.
        prose = scripted_prose_count(text)
        census = (
            f'the addressed text carries {prose} scripted PROSE character(s) — URLs, emails, paths and identifiers are '
            'excluded from the census because their letters are Latin by protocol'
        )
        if prose < MIN_PROSE_SCRIPT_CHARS:
            return (
                f'{census} — which is under the {MIN_PROSE_SCRIPT_CHARS}-character floor a '
                f'{SCRIPT_DOMINANCE_SHARE:.0%} dominance share needs to mean anything: this scope is too short to '
                'establish a script, so the terms could not be checked against it either way'
            )
        return (
            f'{census} — and no writing system carries {SCRIPT_DOMINANCE_SHARE:.0%} of them, so this scope is too '
            'mixed to establish a script: the terms could not be checked against it either way'
        )
    plausible = [t for t in terms if not (term_scripts := scripts_present(t)) or term_scripts & text_scripts]
    if match == 'all':
        impossible = [t for t in terms if t not in plausible]
        if not impossible:
            return ''
        return (
            f'term(s) {", ".join(repr(t) for t in impossible)} are written in '
            f'{", ".join(sorted({s for t in impossible for s in scripts_present(t)}))}, and the addressed text is '
            f'written in {", ".join(sorted(text_scripts))} — under match=all one impossible term makes the whole '
            'conjunction unsatisfiable, so it could never have matched'
        )
    if plausible:
        return ''
    return (
        f'every term is written in {", ".join(sorted({s for t in terms for s in scripts_present(t)}))} while the '
        f'addressed text is written in {", ".join(sorted(text_scripts))} — a case-insensitive substring match across '
        'alphabets can never fire, so the verdict would be vacuous'
    )


def artifact_fact_verdicts(payload: Mapping[str, Any]) -> list[Any] | None:
    """The ``report.sections.fact_verification.verdicts`` rows an artifact publishes, or ``None``.

    ONE reader for the nesting :meth:`FactVerification.as_dict` writes, because consumers outside
    this module re-decide facts off it and a second spelling of that path is how one of them ends
    up reading a section the other does not.

    ``None``, never ``[]``, when the artifact publishes no readable list: the callers report that
    nothing could be re-decided, which is a different fact from a published list that is empty.
    """
    report = payload.get('report')
    sections = report.get('sections') if isinstance(report, Mapping) else None
    section = sections.get('fact_verification') if isinstance(sections, Mapping) else None
    verdicts = section.get('verdicts') if isinstance(section, Mapping) else None
    return verdicts if isinstance(verdicts, list) else None


def _in_scope(unit: CorpusUnit, scope: Sequence[str]) -> bool:
    return not scope or any(fnmatch.fnmatch(unit.unit_id, pattern) for pattern in scope)


def _excerpt_around(haystack: str, needle: str) -> str:
    """A short quote centred on the match, so the evidence shows the sentence, not the file.

    ``needle`` is always a term that already matched, so ``find`` locates it; the guard below
    covers only the empty-needle call a zero-term matcher would make, which the caller rejects
    earlier. Kept because a helper that slices on an unchecked index is a worse trade.
    """
    position = haystack.lower().find(needle.lower())
    if position < 0:
        return haystack[:_EXCERPT_CHARS]
    start = max(0, position - _EXCERPT_CHARS // 3)
    return ('…' if start else '') + haystack[start : start + _EXCERPT_CHARS] + ('…' if start + _EXCERPT_CHARS < len(haystack) else '')


def _addressed_count(fact: ScanFact, units: Sequence[CorpusUnit]) -> int:
    """How many scopes the fact addresses, WITHOUT normalizing any text. See `verify_fact`."""
    return sum(len(unit.scope_labels(fact.field_name)) for unit in units if _in_scope(unit, fact.scope))


class _Scope(NamedTuple):
    """One addressed population: its address, its text, and whether that text is readable at all."""

    label: str
    text: str
    undecodable: bool


def _addressed_scopes(fact: ScanFact, units: Sequence[CorpusUnit]) -> list[_Scope]:
    """Every scope the fact addresses — one per unit, or one per matching field.

    This, not the unit list, is the population a count is taken over: a fact scoped to a catalog
    COLUMN is a claim about the entries in that column, and counting the FILES that hold them
    would answer a different question with the same number.

    The unit's ``undecodable`` flag rides each scope it contributed, because that is the level a
    verdict has to refuse at: a fact addressing ten files of which one is a page of U+FFFD was
    decided over nine, and only the scope knows which.
    """
    return [_Scope(label, text, unit.undecodable) for unit in units if _in_scope(unit, fact.scope) for label, text in unit.searchable(fact.field_name)]


def _matching_scopes(fact: ScanFact, scopes: Sequence[_Scope]) -> list[tuple[str, str]]:
    """The addressed scopes whose text satisfies the fact's term matcher.

    Matching runs over each WHOLE scope with whitespace normalized — never line by line. A claim
    whose wording spans a wrapped markdown sentence is exactly the case a line-scoped matcher
    misses, and a miss here reads as "the corpus does not contain it", which is the verdict.
    """
    hits: list[tuple[str, str]] = []
    for scope in scopes:
        lowered = scope.text.lower()
        found = [t for t in fact.terms if t.lower() in lowered]
        satisfied = len(found) == len(fact.terms) if fact.match == 'all' else bool(found)
        if satisfied:
            hits.append((scope.label, _excerpt_around(scope.text, found[0] if found else '')))
    return hits


def _unverifiable(fact: ScanFact, reason: str, scopes_addressed: int, kind: str) -> FactVerdict:
    return FactVerdict(fact=fact, status=UNVERIFIED_AT_SCAN, reason=reason, scopes_addressed=scopes_addressed, unverified_kind=kind)


def _precedence_note(fact: ScanFact) -> str:
    """Which side to fix when the corpus and the fact disagree. Both sides are always named.

    Three cases, not two. An origin this module does not rank is neither outranked nor a claim to
    have been read off the corpus, and telling its author "you mis-transcribed the corpus" sends
    them to check a document they never read.
    """
    if fact.origin in _OUTRANKED_ORIGINS:
        return f"the corpus OUTRANKS '{fact.origin}' on a corpus-derivable claim (grounding-context contract), so the FACT is what needs fixing, not the corpus"
    if fact.origin == 'corpus':
        return "the fact declares origin 'corpus' — it claims to have been read OFF the corpus that now contradicts it, so one of the two was mis-transcribed"
    return (
        f"the fact declares an origin this checker does not rank ('{fact.origin}'); the corpus is the only side here "
        'with evidence, so treat the fact as unproven until its source is named'
    )


def _declared_check_shape(fact: ScanFact) -> str:
    """What executable half a fact declared, for a reason that names it."""
    parts = ([f'terms {list(fact.terms)}'] if fact.terms else []) + ([f'scope {list(fact.scope)}'] if fact.scope else [])
    return ' and '.join(parts)


def verify_fact(fact: ScanFact, units: Sequence[CorpusUnit]) -> FactVerdict:
    """Decide ONE fact against the corpus."""
    if fact.kind == KIND_BEHAVIOURAL:
        return _unverifiable(
            fact,
            'a behavioural claim — not decidable against corpus content; verify it by probing the agent, not by reading the corpus',
            _addressed_count(fact, units),
            UNVERIFIED_NOT_DERIVABLE,
        )
    if fact.kind == KIND_ASSERTED:
        return _unverifiable(
            fact,
            'asserted, unverified — no static route to check this: the author declared the claim and no executable '
            'check for it, so nothing here decided it either way. Give it a `check:` if reading the corpus settles it; '
            'otherwise it is a premise generation will build on that this scan did not test',
            _addressed_count(fact, units),
            UNVERIFIED_NO_STATIC_ROUTE,
        )
    if fact.kind not in DERIVABLE_KINDS and (fact.terms or fact.scope):
        # An unknown kind that CARRIES A CHECK — terms or a scope — is a check its author meant to
        # run, almost always under a misspelled kind (`absense`). Calling it behavioural filed it
        # with the claims nobody expects the corpus to decide, so the headline never named it. It
        # is a check that COULD NOT EXECUTE: still advisory, but an untested premise by name.
        return _unverifiable(
            fact,
            f'unknown kind {fact.kind!r} on a fact that declares a check ({_declared_check_shape(fact)}) — no check exists '
            f'for this kind, so it could not execute (the kinds this checker decides are {sorted(FACT_KINDS)}); fix the kind',
            _addressed_count(fact, units),
            UNVERIFIED_UNRESOLVABLE,
        )
    if fact.kind not in DERIVABLE_KINDS:
        # ADMITTED AS BEHAVIOURAL, as :func:`load_scan_facts` documents: no check exists for an
        # unknown kind, so nothing is decided and nothing is held. The reason names the kind, so a
        # transposed one (`absense`) reads as exactly that on the verdict row.
        return _unverifiable(
            fact,
            f'unknown kind {fact.kind!r} — admitted as behavioural, not decided (the kinds this checker decides are {sorted(FACT_KINDS)})',
            _addressed_count(fact, units),
            UNVERIFIED_NOT_DERIVABLE,
        )
    if not fact.terms:
        return _unverifiable(
            fact,
            f"kind '{fact.kind}' is corpus-derivable but this fact declares no terms to match, so there is nothing to execute",
            _addressed_count(fact, units),
            UNVERIFIED_UNRESOLVABLE,
        )
    if fact.kind == KIND_COUNT and fact.expected_count is None:
        return _unverifiable(
            fact, "kind 'count' requires expected_count, which this fact does not declare", _addressed_count(fact, units), UNVERIFIED_UNRESOLVABLE
        )

    # Only now is the corpus normalized. The four guards above decide without reading any text,
    # and a behavioural fact — the expected common case — would otherwise normalize the whole
    # corpus to populate one informational integer.
    scopes = _addressed_scopes(fact, units)
    addressed = len(scopes)
    if not scopes:
        where = f'scope {list(fact.scope)}' + (f' / field {fact.field_name!r}' if fact.field_name else '')
        return _unverifiable(
            fact,
            f'the declared {where} addressed NO corpus text, so every verdict over it would be vacuous — an absence '
            'claim over an empty scope is true and proves nothing. Fix the scope: the corpus is addressed by unit ids '
            f'like {[u.unit_id for u in units[:3]]}',
            addressed,
            UNVERIFIED_UNRESOLVABLE,
        )

    if damaged := [s.label for s in scopes if s.undecodable]:
        if len(damaged) == addressed:
            # UNRESOLVABLE for EVERY kind, not just absence. The text a matcher would read is a
            # page of replacement characters, so an existence claim reads false and an absence
            # claim reads true — over bytes nobody decoded.
            return _unverifiable(
                fact,
                f'all {addressed} addressed scope(s) failed to decode ({", ".join(damaged[:_MAX_EVIDENCE])}); their text is '
                'replacement characters, so no literal could match and every verdict over them would be about the decode, '
                'not about the corpus. Re-save the source(s) as UTF-8',
                addressed,
                UNVERIFIED_UNRESOLVABLE,
            )
        if fact.kind == KIND_ABSENCE:
            # Absence is the claim damage can silently satisfy: the destroyed characters ARE the
            # ones the terms are made of. Existence/count still decide on the readable remainder
            # — a hit there is a real hit — so only this arm refuses.
            return _unverifiable(
                fact,
                f'{len(damaged)} of {addressed} addressed scope(s) failed to decode ({", ".join(damaged[:_MAX_EVIDENCE])}), and an '
                'ABSENCE cannot be established over text whose characters were destroyed by the decode — the terms would be '
                'unfindable there whether or not the corpus contains them. Re-save the source(s) as UTF-8',
                addressed,
                UNVERIFIED_UNRESOLVABLE,
            )

    hits = _matching_scopes(fact, scopes)
    evidence = tuple(FactEvidence(unit_id=label, excerpt=text) for label, text in hits[:_MAX_EVIDENCE])
    terms = ', '.join(repr(t) for t in fact.terms)

    def verdict(status: FactStatus, reason: str, *, cite: bool = True) -> FactVerdict:
        """One construction site for all five outcomes, so no arm can forget a field."""
        return FactVerdict(fact=fact, status=status, reason=reason, evidence=evidence if cite else (), scopes_addressed=addressed)

    if fact.kind == KIND_ABSENCE:
        if hits:
            names = ', '.join(label for label, _ in hits[:_MAX_EVIDENCE])
            return verdict(
                CONTRADICTED, f'asserts ABSENCE ({terms}, match={fact.match}) but {len(hits)} corpus scope(s) match it — {names}. {_precedence_note(fact)}'
            )
        # SCRIPT CHECK, and only on this arm. A no-hit absence is the ONE verdict this checker
        # reaches by finding nothing, so it is the one an impossible matcher produces silently —
        # and it is the verdict that mints an adversarial probe carrying "the agent must decline"
        # as graded ground truth. The other kinds fail LOUD under the same mismatch (an existence
        # claim comes back CONTRADICTED, which a human reads).
        # PER SCOPE, at the SAME granularity the matcher ran at. Deciding it over the joined
        # corpus is a granularity bug with the blocking direction: five Hebrew guides among
        # thirty English ones are 14% of the join, so a correct Hebrew fact would be called
        # impossible and the scan would hard-fail on it. The absence is vacuous only when NO
        # addressed scope could have matched — one scope that could is a real place the terms
        # were looked for and not found.
        if reachable := [s for s in scopes if not script_mismatch(fact.terms, fact.match, s.text)]:
            return verdict(VERIFIED, f'none of the {addressed} addressed scope(s) matches {terms} ({len(reachable)} of them could have)', cite=False)
        mismatch = script_mismatch(fact.terms, fact.match, ' '.join(s.text for s in scopes))
        return _unverifiable(
            fact,
            f'the absence would have VERIFIED by finding nothing, but NONE of the {addressed} addressed scope(s) could '
            f"have matched it: {mismatch}. Write check.terms in the corpus's own language(s); an absence nothing could "
            'have matched is not an absence this scan established',
            addressed,
            UNVERIFIED_UNRESOLVABLE,
        )

    if fact.kind == KIND_EXISTENCE:
        if hits:
            return verdict(VERIFIED, f'{len(hits)} of {addressed} addressed scope(s) carry {terms}')
        return verdict(
            CONTRADICTED, f'asserts the corpus contains {terms}, and none of the {addressed} addressed scope(s) does. {_precedence_note(fact)}', cite=False
        )

    if fact.kind == KIND_COUNT:
        if len(hits) == fact.expected_count:
            return verdict(VERIFIED, f'exactly {fact.expected_count} of {addressed} addressed scope(s) match {terms}')
        return verdict(CONTRADICTED, f'asserts {fact.expected_count} matching scope(s) for {terms}; the corpus has {len(hits)}. {_precedence_note(fact)}')

    # Unreachable while DERIVABLE_KINDS is derived from the branches above — and that derivation
    # is what makes it unreachable, so this is the guard that keeps it so. A kind added to the
    # frozenset without a branch would fall through to COUNT semantics and contradict
    # everything against expected_count=None.
    return _unverifiable(fact, f'kind {fact.kind!r} is declared derivable but no check is implemented for it', addressed, UNVERIFIED_UNRESOLVABLE)


def verify_facts(facts: Iterable[ScanFact], units: Sequence[CorpusUnit]) -> FactVerification:
    """Decide every fact. Order is preserved so a report reads in the author's own order."""
    return FactVerification(verdicts=[verify_fact(f, units) for f in facts])


#: The typed predicate that says the customer's OWN inputs state a value incompatibly — the
#: cross-source conflict class. It is a STATEMENT ABOUT THE INPUTS, not about anything the
#: agent or a scenario does: no rewrite on either side can satisfy both documents, so a
#: consumer that treats a fact carrying it as a defect to repair spends rounds re-deriving it.
CONFLICTS_ACROSS_SOURCES = 'conflicts_across_sources'


def parse_agent_facts(prd_markdown: str | None) -> list[dict[str, Any]]:
    """The typed ``## Agent facts`` entries from a PRD (``kind`` × ``subject`` × ``predicate``,
    each with ``evidence`` + ``source``), or ``[]`` when it carries none.

    The block ends at a fence LINE: a column-0 ```` ``` ```` on a line of its own. A fence
    quoted inside an evidence string or block scalar is indented YAML content, never the end.

    ONE parser, shared by every consumer of the block. The scan authors it and lints it only
    advisorily, so a malformed one must degrade to "no facts" rather than fail a job that would
    otherwise have shipped — a raising parser here would take down eval generation over a
    hand-edited PRD.
    """
    if not prd_markdown:
        return []
    section = re.search(r'^##\s+Agent facts\s*$(.*?)(?=^##\s|\Z)', prd_markdown, re.S | re.M)
    if not section:
        return []
    fence = re.search(r'^```(?:yaml)?[ \t]*\n(.*?)^```[ \t]*$', section.group(1), re.S | re.M)
    if not fence:
        return []
    import yaml  # noqa: PLC0415 - lazy: this module is otherwise stdlib-only
    from plurai.scan_contract._safe_yaml import safe_yaml_load  # noqa: PLC0415

    try:
        block = safe_yaml_load(fence.group(1))
    except yaml.YAMLError:
        return []
    facts = block.get('facts') if isinstance(block, dict) else None
    return [f for f in (facts or []) if isinstance(f, dict)]


#: The ``tool_surface`` predicate that says the agent's OWN shipped config binds a retrieval
#: capability's source (an index, a KB / RAG-config id, a collection) to empty or unset. A DEFECT
#: HYPOTHESIS, never a boundary: with no source bound there is no knowledge base that could
#: honestly lack an answer, so a decline on an in-contract question is a delivery failure, not an
#: honest gap. Deliberately NOT a member of ``probes.CAPABILITY_PREDICATES`` — an absence probe
#: grades a DECLINE as correct, which would grade this defect as the right behaviour.
#: The plugin's ``lint_prd.py`` ``_FACT_PREDICATES`` carries a copy of this spelling: if the two
#: drift, the scan's fact lints clean there and this reader matches nothing, so no delivery judge
#: is mandated.
SOURCE_UNBOUND = 'source_unbound'


def unbound_capability_facts(prd_markdown: str | None) -> dict[str, list[dict[str, Any]]]:
    """``{tool: [fact, ...]}`` for every ``tool_surface`` / ``source_unbound`` fact in the PRD,
    grouped by the tool the subject names (``<tool>[.<arg>]``), in the order the PRD lists them.

    ``{}`` when the PRD carries none — which is every agent whose scan found no empty binding, and
    what keeps every consumer of this reading inert for them. Read through :func:`parse_agent_facts`,
    the one parser of the block.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in parse_agent_facts(prd_markdown):
        if str(fact.get('kind') or '') != 'tool_surface' or str(fact.get('predicate') or '') != SOURCE_UNBOUND:
            continue
        tool = str(fact.get('subject') or '').split('.', 1)[0].strip()
        if tool:
            grouped.setdefault(tool, []).append(fact)
    return grouped


def _verdict_predicate(row: Mapping[str, Any]) -> str:
    """The predicate a published verdict declares, off its OWN typed fields.

    ``FactVerdict.as_dict`` does not publish ``predicate``, but the projection mints
    ``fact_id`` as ``<subject>.<predicate>`` (``items.id.closed_set``,
    ``RENDER_ACTIONS.omits``), so the suffix IS the predicate — derived, never guessed from the
    claim prose. A ``predicate`` key is honoured first so an artifact that starts publishing one
    needs no change here.
    """
    if predicate := str(row.get('predicate') or '').strip():
        return predicate
    fact_id = str(row.get('fact_id') or '').strip()
    subject = str(row.get('subject') or '').strip()
    return fact_id[len(subject) + 1 :] if subject and fact_id.startswith(f'{subject}.') else ''


def cross_source_conflict_subjects(prd_markdown: str | None = None, scan_artifact: Any = None) -> dict[str, str]:
    """``{subject: why it is disputed}`` for every fact the typed ledger types as a CONFLICT
    BETWEEN THE CUSTOMER'S OWN SOURCES — keyed on the typed ``subject``, never on prose.

    Two typings count, and they are the two the ledger has:

    * ``predicate: conflicts_across_sources`` — the author measured the value as stated
      incompatibly by two of the customer's documents and said so up front.
    * ``status: contradicted`` on a published verdict — this module's own arbitration found the
      corpus refuting the document that declared the fact, which is the same disagreement
      reached from the other end.

    The point of naming these subjects is that NOTHING DOWNSTREAM CAN REPAIR THEM: a consumer
    holding a conflicted subject against a scenario (or an agent) is grading a document dispute,
    and the only fix is on the operator's desk. Callers use this to route such a finding to the
    operator instead of spending a repair round on it.
    """
    conflicted: dict[str, str] = {}
    for fact in parse_agent_facts(prd_markdown):
        if str(fact.get('predicate') or '').strip() != CONFLICTS_ACROSS_SOURCES:
            continue
        if subject := str(fact.get('subject') or '').strip():
            evidence = ' '.join(str(fact.get('evidence') or '').split())
            conflicted.setdefault(subject, evidence or f'typed `## Agent facts` entry {subject}.{CONFLICTS_ACROSS_SOURCES}')
    rows = artifact_fact_verdicts(scan_artifact) if isinstance(scan_artifact, Mapping) else None
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        subject = str(row.get('subject') or '').strip()
        if not subject:
            continue
        if _verdict_predicate(row) != CONFLICTS_ACROSS_SOURCES and str(row.get('status') or '') != CONTRADICTED:
            continue
        claim = ' '.join(str(row.get('claim') or '').split())
        conflicted.setdefault(subject, claim or f'scan verdict {row.get("fact_id") or subject}')
    return conflicted
