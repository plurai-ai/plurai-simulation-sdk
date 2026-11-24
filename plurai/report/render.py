"""The deterministic markdown renderings of a run: ``report.md``, the job summary body and the PR comment body.

The same input renders the same bytes: no clock, no network, no model, no environment. What the report cannot know it
takes as plain input in :class:`ReportOptions`: main's run, the repository's visibility, notes, the token's expiry,
today's date and, for a run that ended at a stop, the :class:`Stopped` heading and fix. The text a reader sees
says "scenario × policy" and "suite ids"; "cell" and "pin" appear only in identifiers.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Final, Literal

from plurai import client_handshake, exit_codes
from plurai.report.outcome import CLASS_ORDER, INVALID_CLASS, REGRESSION_SCENARIO_PREFIX, REGRESSION_TAG, NotMeasuredClass
from plurai.report.run import (
    CAUSES,
    CREDIT_CODES,
    NEEDS_REDEPLOY,
    QUARANTINED,
    REPO_SHAPE,
    TIERS,
    UNKNOWN_CAUSE,
    Cell,
    Credits,
    Ignored,
    NotRun,
    OverrideNote,
    Run,
    cause_parts,
    classify,
    scrub_surrogates,
)
from plurai.report.tally import QUARANTINE_CAP_PERCENT, Tally, newly_quarantined, tally

#: GitHub caps a job summary at 1 MiB.
SUMMARY_LIMIT_BYTES = 1024 * 1024

#: GitHub rejects a comment body over this many characters.
COMMENT_LIMIT_CHARS = 65_536

#: The room the action leaves for the suite diff it appends to the comment (``suite-diff-comment.md``) and to the summary
#: (``suite-diff.md``). The diff renderer keeps each file inside its share, and the comment and the summary are cut to the rest, so
#: the two joined never pass GitHub's limits. The slack covers the separator the action writes between the two.
SUITE_DIFF_COMMENT_CHARS = 8_000
SUITE_DIFF_SUMMARY_BYTES = 256 * 1024
_APPEND_SLACK = 64

#: The Checks API rejects a check run's text over this many characters, and ``report.md`` is that text.
CHECK_RUN_LIMIT_CHARS = 65_535

#: The sticky PR comment is found again by this marker and edited in place.
COMMENT_MARKER = '<!-- plurai-simulation-report -->'

#: The mark of one trial: violated, held, not applicable (a deferred trial is not the policy's to judge) and not measured.
TRIAL_MARKS: Final[dict[str, str]] = {
    'violated': '❌',
    'clear': '✅',
    'not_applicable': '➖',
    'deferred': '➖',
    'invalid_infra': '⏸',
    'invalid_agent': '⏸',
    'invalid_suite': '⏸',
    'invalid_credits': '⏸',
}

#: The marker in front of each tier.
TIER_MARKS: Final[dict[str, str]] = {'catastrophic': '🔴', 'failing': '🟠', 'standard': '🟡'}

#: The statuses a scenario × policy shows, in column order: its icon, the words of a summary line and the column header.
_STATUSES: Final = (
    ('❌', 'failed', '❌ Failed'),
    ('⚠️', 'unstable', '⚠️ Unstable'),
    ('✅', 'passed', '✅ Passed'),
    ('⏸', 'not measured', '⏸ Not measured'),
    ('➖', 'N/A', '➖ N/A'),
)
#: The statuses the overview columns and the summary lines count. N/A is neither: a scenario's N/A policies are one folded list.
_COUNTED: Final = _STATUSES[:-1]
#: The icon a scenario takes is that of its worst scenario × policy, in this order; scenarios are listed in it too.
_STATUS_RANK: Final = ('❌', '⚠️', '⏸', '✅', '➖')

#: A scenario title or policy description from the suite is cut to this many characters.
_NAME_LIMIT = 90

#: Notes no reader acts on: they go in the closed "Run details" section, not the visible notes.
_QUIET_NOTES: Final = ('token expiry unknown', 'the credit state was not checked')

#: Under this many days to expiry the token line is also a note at the top of the report.
TOKEN_EXPIRY_NOTE_DAYS = 14

#: Main's run older than this many days is named with its age: a scheduled run that stopped leaves the baseline to age.
MAIN_AGE_NOTE_DAYS = 7

#: How to get a main baseline: the default triggers never run main on their own.
_RUN_MAIN: Final = 'run the workflow on main (Actions → Run workflow), or turn on the `push` trigger'

#: The suite ids a base commit can name, in the order the report lists them.
_SUITE_KINDS: Final = ('scenarios', 'evals', 'env', 'corpus')

#: What a scenario × policy with no tier shows where its tier would be.
NO_TIER = 'no tier'

Visibility = Literal['public', 'private', 'internal']

_REASONING_LIMIT = 140

_SHORT_ID = 8
#: A policy id longer than this is shown with its middle cut out; ``run.json`` and ``junit.xml`` keep the whole id.
_POLICY_ID_LIMIT = 40
_POLICY_ID_HEAD = 18
_TOO_FEW_VALID = 'fewer than 2 valid trials'
_WITHHELD = 'Judge text is withheld; set `include_content: true` in the workflow to show it.'
_WEB_ADDRESS = re.compile(r'https?://[^\s<>`\\\[\]*|]+')
_SELF_CONSISTENCY = re.compile(r'^\[self-consistency[^\]]*\]\s*')
#: A lowercase plain word, not an identifier such as ``orders_lookup``, which keeps its spelling when it opens a sentence.
_PLAIN_WORD = re.compile(r'[a-z]+\b')
_RUBRIC_CODE = r'[A-Z]\d{1,2}'
_RUBRIC_JOIN = r'(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*)'
#: The rubric codes (``F7:``, ``S2 and S7:``, ``S2, S5:``, ``H1 —``) that open a judge's reason, and a list of them in
#: parentheses (``(H1)``) anywhere in it.
_RUBRIC_PREFIX = re.compile(rf'^{_RUBRIC_CODE}(?:{_RUBRIC_JOIN}{_RUBRIC_CODE})*(?:\s*[:—–]|\s+-(?=\s))\s*')
_RUBRIC_PAREN = re.compile(rf'\s*\({_RUBRIC_CODE}(?:{_RUBRIC_JOIN}{_RUBRIC_CODE})*\)')
#: Where a code evaluator's finding turns from a sentence into the expression that failed.
_EVALUATOR_EXPRESSION = re.compile(r'\s*\|\s*FAILING ELEMENT:.*', re.DOTALL)
#: A Python list of quoted names, as an evaluator prints it.
_QUOTED_LIST = re.compile(r"\[((?:'[^']*'(?:,\s*)?)+)\]")
#: The finding of the fact-coverage check: the keys of the facts the answer never states.
_UNVOICED_FACTS = re.compile(r"^unvoiced facts:\s*\[((?:'[^']*'(?:,\s*)?)+)\]")
#: The fact the check printed as the failing element: its key and its value as a Python string literal. The check cuts its
#: output, so the value is there only for the first fact and only when the cut leaves it whole.
_SAW_FACT = re.compile(r"""saw f=\{'key': '([^']*)', 'kind': '[^']*', 'value': ('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""")
#: A fact value longer than this is left out of the sentence.
_FACT_VALUE_LIMIT = 40
_SUMMARY_TAG = re.compile(r'<summary>(.*?)</summary>', re.DOTALL)
#: The PR comment lists at most this many scenario × policy; the job summary lists them all.
_COMMENT_ROWS = 5
#: Each reason in the PR comment is cut to this many characters.
_COMMENT_WHY_LIMIT = 160
_MARKS_LEGEND = "First icon: this run's result. Trials: one mark per trial (❌ violated · ✅ held · ⏸ not measured · ➖ did not come up)."
_CUT_POINTER = 'cut to fit the job summary limit; the full list is in report.md'


#: What follows the verdict mark in the heading of a cancelled run (the words the action uses when it wrote no report), and what to do.
CANCELLED_LEAD: Final = 'not measured: the run was cancelled'
CANCELLED_FIX: Final = 'The job was cancelled, for example by a newer push. Run the job again to measure it.'

#: The remedy for a committed id that is missing or does not link to the others, and the one for an id the backend does not hold.
IDS_FIX: Final = 'commit ids that exist on the backend and link to each other.'
NOT_ON_BACKEND_FIX: Final = (
    f'not found on this backend: check that `PLURAI_TOKEN` is for the org that owns the suite, and that the box still holds it. Otherwise {IDS_FIX}'
)


@dataclass(frozen=True)
class Stopped:
    """A run that ended at a stop, before the first thread or after the results it keeps in ``run.json``: the heading it earns
    and the paragraph that says what went wrong and what to do.

    Build one with a constructor below; the renderers then show the heading, the paragraph and the footer, and no counts or
    tables. The heading is also the check's title, which GitHub shows as plain text, so ``reason`` is a few plain words with
    no ids, paths or markup; the values and the fix go in ``fix``, markdown the constructors have already escaped.
    """

    #: What follows the verdict mark in the heading, such as ``not measured: no secrets (fork PR)``.
    reason: str
    fix: str
    #: The row of :mod:`plurai.exit_codes` the run exits with: ``REFUSED``, ``CREDENTIALS``, ``UPGRADE_REQUIRED`` or ``NOT_MEASURED``.
    exit_code: int = exit_codes.NOT_MEASURED
    #: The problem with its values, as plain text for the job log, when ``reason`` leaves them out.
    detail: str = ''

    @property
    def log_line(self) -> str:
        """The one line the job log gets: the heading's words, then the detail."""
        return f'{self.reason} (exit {self.exit_code})' + (f': {" ".join(self.detail.split())}' if self.detail else '')

    @classmethod
    def no_secrets(cls, kind: Literal['fork PR', 'Dependabot', 'PLURAI_TOKEN is not set']) -> Stopped:
        """The job has no ``PLURAI_TOKEN``. The caller names why when it knows: a pull request from a fork, or one Dependabot
        opened. A caller that knows only that the token is missing, as ``plurai ci run`` does, says ``PLURAI_TOKEN is not set``."""
        if kind == 'PLURAI_TOKEN is not set':
            fix = 'Nothing ran: `PLURAI_TOKEN` is not set. Pass the service-account key in the workflow, as a secret that this run can read.'
        elif kind == 'fork PR':
            fix = (
                'Nothing ran: the workflow has no `PLURAI_TOKEN` on a pull request from a fork. '
                'A maintainer can push the branch to this repository to measure it.'
            )
        else:
            fix = (
                "Nothing ran: Dependabot's secrets do not include `PLURAI_TOKEN`. "
                'Give Dependabot the secret with `gh secret set --app dependabot PLURAI_TOKEN`.'
            )
        return cls(f'not measured: no secrets ({kind})', fix)

    @classmethod
    def empty_env(cls, name: str) -> Stopped:
        """A variable the agent declares in ``agent.env`` is empty in the job's environment."""
        fix = (
            f'Nothing ran: the agent declares {_code(name)} in `agent.env` and the job has it empty. '
            'Set it as a secret or a variable and pass it in the workflow `env:` block.'
        )
        return cls(f'not measured: no secrets ({name} empty)', fix)

    @classmethod
    def gateway_402(cls, code: str | None, resets_at: str | None = None) -> Stopped:
        """The LLM gateway answered 402. With one of :data:`CREDIT_CODES` it is a credits stop, never retried; with any
        other body nothing says credits are the cause, so it is an infra stop."""
        if code in CREDIT_CODES:
            return cls(
                f'not measured: credits ({_credits_detail(Credits(code, resets_at))})',
                f'Nothing ran. {CAUSES["credits"].next_step[0].upper()}{CAUSES["credits"].next_step[1:]}.',
            )
        return cls.not_measured('infra', CAUSES['gateway_refused'].words, f'{CAUSES["gateway_refused"].next_step}.')

    @classmethod
    def for_cause(cls, cause: str) -> Stopped:
        """A step before the first thread, or the whole run, ended on ``cause``, a key of ``CAUSES`` such as ``needs_redeploy`` or
        ``gateway_timeout``. A cause the SDK does not know is infra and is named."""
        known = CAUSES.get(cause)
        if known is None:
            return cls.not_measured('infra', f'{UNKNOWN_CAUSE.words}: {cause}', f'{UNKNOWN_CAUSE.next_step}.')
        return cls.not_measured(known.cls, known.words, f'{known.next_step}.')

    @classmethod
    def cancelled(cls) -> Stopped:
        """A signal cancelled the job before any scenario × policy was measured."""
        return cls(CANCELLED_LEAD, f'Nothing was measured. {CANCELLED_FIX}')

    @classmethod
    def ids_unavailable(cls, why: str, detail: str, *, fix: str = IDS_FIX) -> Stopped:
        """A committed id is missing on the backend or does not link to the other committed ids: ``why`` in a few words for the
        heading, ``detail`` with the ids, ``fix`` the remedy when the default does not say enough. A new run cannot fix it, so the
        class is suite."""
        return cls.not_measured('suite', why, fix, detail=detail)

    @classmethod
    def not_on_backend(cls, kind: str, artifact_id: str) -> Stopped:
        """A committed id is not found on the backend the token reaches: the suite may be wrong, and so may the token's org or a
        box that was rebuilt empty."""
        return cls.ids_unavailable(
            f'the committed {kind} is not on the backend', f'the committed {kind} {artifact_id} is not on the backend', fix=NOT_ON_BACKEND_FIX
        )

    @classmethod
    def too_many_quarantined(cls, quarantined: int, gating: int) -> Stopped:
        """More than the cap's share of the selected gating scenarios are quarantined as unsolvable: the suite, not the agent,
        is what failed."""
        why = f'too many quarantined, {quarantined} of {gating} gating scenarios'
        return cls.not_measured('suite', why, f'{CAUSES[QUARANTINED].next_step}.')

    @classmethod
    def refused(cls, heading: str, detail: str) -> Stopped:
        """The run declined to start on what it checked first. ``heading`` names the problem in a few plain words; ``detail``
        is the whole message, the files and values it names and the fix, with code and paths in single backticks."""
        return cls(f'refused: {heading}', f'Nothing ran. {_inline(_sentence(detail))}', exit_codes.REFUSED, detail)

    @classmethod
    def token_rejected(cls, repo: str | None = None) -> Stopped:
        """The backend answered 401: the token is expired, revoked or wrong."""
        if repo and not REPO_SHAPE.fullmatch(repo):
            raise ValueError(f'repo is not an owner/name: {repo!r}')
        mint = _code(f'plurai ci token --github {repo or "<owner>/<repo>"}')
        return cls(
            'token rejected',
            f'Nothing ran: the backend rejected `PLURAI_TOKEN`, which is expired, revoked or wrong. Mint a new key with {mint}.',
            exit_codes.CREDENTIALS,
        )

    @classmethod
    def token_refused(cls, route: str) -> Stopped:
        """The backend answered 403 on ``route``: the token is valid but may not do this."""
        return cls(
            f'token refused for {route}',
            'Nothing ran: the backend refused `PLURAI_TOKEN` for that route. Use a key minted for this repository with `plurai ci token`.',
            exit_codes.CREDENTIALS,
        )

    @classmethod
    def upgrade_required(cls, refusal: client_handshake.UpgradeRefusal, *, after_results: bool = False) -> Stopped:
        """The backend answered 426: it refuses this SDK version on every call. The fix is the handshake module's own text,
        so the summary, the comment and the job log say the same thing. ``after_results`` is for a refusal that ended a run that
        had already measured something, which ``run.json`` keeps."""
        lead = 'Nothing ran.'
        if after_results:
            lead = 'The run stopped at the refusal; what was measured before it is in `run.json`.'
        return cls(
            'upgrade required: the backend refuses this SDK version',
            f'{lead}\n\n{_fenced(client_handshake.upgrade_message(refusal))}',
            exit_codes.UPGRADE_REQUIRED,
        )

    @classmethod
    def not_measured(cls, name: NotMeasuredClass, why: str, fix: str, *, detail: str = '') -> Stopped:
        """A step ended the run for an infra, agent, suite or credits cause. ``why`` is the plain-words cause, short
        and with no values, and ``fix`` the remedy that cause calls for, a sentence ending in a full stop: the class alone does
        not decide it (a committed id the backend does not have is not fixed by running again). ``detail``, the ids or the
        error text, goes before the fix, with code in single backticks."""
        said = f'{_inline(_sentence(detail))} ' if detail else ''
        return cls(f'not measured: {name} ({why})', f'Nothing ran. {said}{fix[0].upper()}{fix[1:]}', detail=detail)


@dataclass(frozen=True)
class ReportOptions:
    """What the renderers take besides the run."""

    #: Main's last run, for the "vs main" line. Used only when it scored the same four suite ids under the same schema.
    main: Run | None = None
    #: ``include_content``: ``True`` or ``False`` decides; ``None`` is auto, on only for a private repository.
    include_content: bool | None = None
    visibility: Visibility | None = None
    #: Why ``main`` is not given although the caller looked for it (an unreadable file, another schema), for the note. Words
    #: only: the runner's file paths are not for the reader.
    main_error: str | None = None
    #: The committed ids at the PR's base commit (``scenarios``, ``evals``, ``env``, ``corpus``), when the caller read them. They
    #: differing from the ids of ``main`` says main's suite changed since its last run.
    base_ids: Mapping[str, str] | None = None
    #: Config and infra remarks the caller knows. They are listed and never counted. Markdown the caller composed: the
    #: renderer keeps it as given, so text from the backend goes in escaped.
    notes: Sequence[str] = ()
    artifacts: Sequence[str] = ()
    #: Where the whole run is shown (the job's page): the PR comment links its "job summary" phrases there. A plain ``http(s)`` address.
    run_url: str | None = None
    #: When the CI token expires, and today's date to count the days from. Both or neither.
    token_expires: date | None = None
    today: date | None = None
    #: Set when the run ended at a stop: every rendering then shows this instead of counts and tables.
    stopped: Stopped | None = None
    #: The directory ``plurai ci run`` is run in, for the reproduce line.
    workdir: str = 'plurai_sim'


def resolve_include_content(setting: bool | None, visibility: Visibility | None) -> bool:
    """Whether judge text goes into the renderings: an explicit setting wins; auto is on only for a ``private`` repository.

    An ``internal`` repository is readable by a whole enterprise and is handled like a public one. An unknown visibility
    under auto leaves the text out.
    """
    return setting if setting is not None else visibility == 'private'


_MARKDOWN_SPECIAL = re.compile(r'([\\`*|\[\]<>])')
#: What GitHub turns into a link or a notification: ``@user`` and ``@org/team`` at a word start, and ``#123`` or ``org/repo#123``.
_MENTION = re.compile(r'(?<!\w)@(?=\w)')
_ISSUE_REFERENCE = re.compile(r'#(?=\d)')
#: Zero width space: invisible, and it keeps the two characters of a mention or an issue reference from reading as one token.
_BREAK = '\u200b'
_BACKTICK_RUN = re.compile('`+')


def _plain(text: str) -> str:
    """``text`` with every character that starts markup, a table column or HTML escaped (the backslash first), and a mention or
    an issue reference broken so GitHub neither notifies the person nor links the issue."""
    escaped = _MARKDOWN_SPECIAL.sub(r'\\\1', text.replace('&', '&amp;'))
    return _ISSUE_REFERENCE.sub(f'#{_BREAK}', _MENTION.sub(f'@{_BREAK}', escaped))


def _md(text: str) -> str:
    """``text`` as one line of inline markdown that shows as written: lone surrogates replaced, whitespace and newlines
    folded, every character that starts markup, a table column or HTML escaped, and ``@mentions`` and ``#123`` references
    defused (they would ping a user or link an issue from a PR comment)."""
    return _plain(' '.join(scrub_surrogates(text).split()))


_CODE_SPAN = re.compile(r'`([^`]+)`')


def _inline(text: str) -> str:
    """``text``, plain words with code in single backticks, as one line of inline markdown: each code span stays a span,
    rebuilt by :func:`_code`, and the words between are escaped as :func:`_md` escapes them, so a value inside either cannot
    start markup."""
    parts = _CODE_SPAN.split(' '.join(scrub_surrogates(text).split()))
    return ''.join(_code(part) if index % 2 else _plain(part) for index, part in enumerate(parts))


def _sentence(text: str) -> str:
    """``text`` as a sentence: a first plain word raised to a capital (a file or key name such as ``ci.yaml`` keeps its
    spelling), a full stop added when it ends without one."""
    text = text.strip()
    first = text.split(' ', 1)[0].rstrip(':,;')
    if first.isalpha() and first.islower():
        text = text[0].upper() + text[1:]
    return text if text.endswith(('.', '!', '?')) else f'{text}.'


def _code(text: str) -> str:
    """``text`` as one inline code span whose delimiter is longer than any run of backticks inside it, so nothing in it can end
    the span; markup inside a span is literal."""
    clean = ' '.join(scrub_surrogates(text).split())
    if not clean:
        return '` `'
    fence = '`' * (1 + max((len(run) for run in _BACKTICK_RUN.findall(clean)), default=0))
    # CommonMark strips one space from each side of a span that has one on both, so a backtick at an edge is kept apart from
    # the fence by that pair of spaces.
    pad = ' ' if clean[0] == '`' or clean[-1] == '`' else ''
    return f'{fence}{pad}{clean}{pad}{fence}'


def _fenced(text: str) -> str:
    """``text`` in a code fence longer than any run of backticks inside it, so nothing in it can close the fence or the
    ``<details>`` block around it."""
    clean = scrub_surrogates(text).strip('\n')
    fence = '`' * max([3, *(len(run) + 1 for run in _BACKTICK_RUN.findall(clean))])
    return f'{fence}text\n{clean}\n{fence}'


def _words(identifier: str) -> str:
    """An id as words: without the ``regression_`` prefix (the scenario block says it once), underscores as spaces, sentence
    case."""
    text = ' '.join(identifier.removeprefix(REGRESSION_SCENARIO_PREFIX).replace('_', ' ').split()) or identifier
    return text[:1].upper() + text[1:]


def _given_name(names: Mapping[str, str], identifier: str) -> str:
    given = ' '.join(scrub_surrogates(names.get(identifier, '')).split())
    return _cut(given, _NAME_LIMIT) if given else _words(identifier)


def scenario_name(run: Run, scenario: str) -> str:
    """The scenario as a reader names it: its title from the suite, else its id in words. Plain text, unescaped."""
    return _given_name(run.scenario_titles, scenario)


def policy_name(run: Run, policy: str) -> str:
    """The policy as a reader names it: its short description from the suite, else its id in words. Plain text, unescaped."""
    return _given_name(run.policy_descriptions, policy)


def _labelled(name: str, identifier: str) -> str:
    """A table cell: the name, and the id in small code under it. A table splits its row at a ``|`` even inside a code span,
    so the id's pipes are escaped; the table drops the backslash before the span is read."""
    return f'{_md(name)}<br><sub>{_code(identifier).replace("|", chr(92) + "|")}</sub>'


def _short_policy_id(identifier: str) -> str:
    """A long policy id with its middle cut out: its head says where it comes from and its tail keeps the part that tells it
    from its siblings. Short ids stay whole."""
    if len(identifier) <= _POLICY_ID_LIMIT:
        return identifier
    return f'{identifier[:_POLICY_ID_HEAD]}…{identifier[-(_POLICY_ID_LIMIT - _POLICY_ID_HEAD - 1) :]}'


def _name(run: Run, cell: Cell) -> str:
    """``scenario · policy`` in words, escaped."""
    return _md(f'{scenario_name(run, cell.scenario)} · {policy_name(run, cell.policy)}')


def status_icon(cell: Cell) -> str:
    """The icon of the cell's outcome; a deferred cell is N/A."""
    return {'failed': '❌', 'unstable': '⚠️', 'passed': '✅', 'not_measured': '⏸'}.get(cell.outcome.outcome, '➖')


def tier_text(cell: Cell) -> str:
    """``🔴 catastrophic``: the severity with its marker, then the other tags, and ``advisory`` for a cell that does not gate.
    ``regression`` is no tag here: the scenario block says it once. Escaped."""
    tier = next((tag for tag in cell.tags if tag in TIERS), None)
    parts = [_tier_label(tier) if tier else NO_TIER, *(tag for tag in cell.tags if tag not in TIERS and tag != REGRESSION_TAG)]
    if cell.outcome.level == 'advisory':
        parts.append('advisory')
    return _md(' · '.join(parts))


def _tier_label(tier: str) -> str:
    """``🔴 catastrophic``: a severity with its marker."""
    return f'{TIER_MARKS[tier]} {tier}'


def _listed(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b, and c``."""
    return ' and '.join(items) if len(items) < 3 else f'{", ".join(items[:-1])}, and {items[-1]}'


def _fail_on_text(fail_on: Iterable[str]) -> str:
    """What the run's ``fail_on.tags`` make gating, in words: the severities that fail the check anywhere, the regression
    scenarios when ``regression`` is listed, and what the other severities count for.

    ``regression`` is a scenario property, not a severity, so it reads as "every policy in a regression scenario"; a list
    that names every severity reads as "every severity", which already covers every regression scenario's tiered policies.
    """
    tags = set(fail_on)
    regression = REGRESSION_TAG in tags
    gating = [_tier_label(t) for t in TIERS if t in tags]
    advisory = [_tier_label(t) for t in TIERS if t not in tags]
    if not advisory:
        parts = ['every severity']
    elif regression:
        parts = [*gating, 'every policy in a regression scenario']
    else:
        parts = gating
    if not parts:
        why = 'is empty' if not tags else 'names no severity and not `regression`'
        return f'This run fails the check on no policy: `fail_on.tags` {why}.'
    lead = f'This run fails the check on: {_listed(parts)}.'
    if not advisory:
        return lead
    rest = 'count only in regression scenarios' if regression else 'are advisory and never fail the check'
    return f'{lead} {_listed(advisory)} policies {rest}.'


def _regression_line(fail_on: Iterable[str]) -> str:
    """The line a ``regression_*`` scenario opens with: whether its regression makes every policy gating in this run."""
    if REGRESSION_TAG in set(fail_on):
        return 'Regression scenario: every policy can fail the check.'
    return 'Regression scenario: `fail_on.tags` leaves out `regression`, so its policies fail the check by severity alone.'


def _cut(text: str, limit: int) -> str:
    """``text`` within ``limit`` characters: cut at the last word boundary that fits, never inside a word, and end with ``…``.

    A single word longer than the limit is cut where it stands.
    """
    if len(text) <= limit:
        return text
    head = text[: limit - 1]
    if not text[limit - 1].isspace() and not head[-1:].isspace():
        boundary = max(head.rfind(' '), head.rfind('\n'), head.rfind('\t'))
        head = head[:boundary] if boundary > 0 else head
    return head.rstrip(' \t\n,;:–—-') + '…'


def _first_sentence(text: str) -> str:
    """The first sentence of ``text`` without a ``[self-consistency …]`` prefix, cut near :data:`_REASONING_LIMIT` characters."""
    text = _SELF_CONSISTENCY.sub('', text).strip()
    match = re.search(r'[.!?](?=\s|$)', text)
    return _cut(text[: match.end()] if match else text, _REASONING_LIMIT)


def _and_list(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else f'{", ".join(items[:-1])} and {items[-1]}'


def _unvoiced_facts_sentence(text: str) -> str | None:
    """``The answer never states refund window (14 days)`` for the fact-coverage check's ``unvoiced facts: [...]``
    finding, else ``None``. The value comes from the failing element the check prints after the finding, when it is whole and
    short; the other facts are named without one."""
    match = _UNVOICED_FACTS.match(text)
    if match is None:
        return None
    values: dict[str, str] = {}
    for key, literal in _SAW_FACT.findall(text):
        try:
            value = ' '.join(str(ast.literal_eval(literal)).split())
        except (ValueError, SyntaxError):
            continue
        if value and len(value) <= _FACT_VALUE_LIMIT:
            values.setdefault(key, value)
    facts = [f'{key.replace("_", " ")}{f" ({values[key]})" if key in values else ""}' for key in re.findall(r"'([^']*)'", match.group(1))]
    return f'The answer never states {_and_list(facts)}'


def plain_sentence(text: str) -> str:
    """A judge's reason as one plain sentence for a reader: without a rubric code (``F7:``, ``(H1)``), without the
    expression a code evaluator prints after its finding, with a list of quoted names written as names and the fact-coverage
    finding said in words. The raw text belongs in the judge reasoning."""
    text = _SELF_CONSISTENCY.sub('', text).strip()
    if (facts := _unvoiced_facts_sentence(text)) is not None:
        return facts
    text = _EVALUATOR_EXPRESSION.sub('', text)
    coded = _RUBRIC_PREFIX.match(text) is not None
    text = _RUBRIC_PAREN.sub('', _RUBRIC_PREFIX.sub('', text))
    text = _QUOTED_LIST.sub(lambda m: ', '.join(re.findall(r"'([^']*)'", m.group(1))), text)
    return text[:1].upper() + text[1:] if coded and _PLAIN_WORD.match(text) else text


_TURN_NUMBER = r'\d+(?!\.?\d)'
_TURN_SEPARATOR = r'(?:\s*,\s*(?:(?:and|or)\s+)?|\s*[&–-]\s*|\s+(?:and|or)\s+)'
_TURN_REFERENCE = re.compile(rf'\b(turns?\s+){_TURN_NUMBER}(?:{_TURN_SEPARATOR}{_TURN_NUMBER})*', re.IGNORECASE)
_DIGITS = re.compile(r'\d+')


def one_based_turns(text: str) -> str:
    """``text`` with each turn reference ("Turn 2", "turns 2 and 4", "turns 0, 2") counted from 1, the way the tree and the
    turns beside a finding count. A judge writes turns from 0, as ``run.json`` records them. Only the numbers a "turn" or
    "turns" introduces move; other numbers, and words that contain "turn", stay."""

    def shift(match: re.Match[str]) -> str:
        return match.group(1) + _DIGITS.sub(lambda number: str(int(number.group()) + 1), match.group()[len(match.group(1)) :])

    return _TURN_REFERENCE.sub(shift, text)


def _turns_text(turns: Sequence[int]) -> str:
    """`` (turns 1, 4)``: the turns a finding points at, counted from 1 as a reader counts them (the run records them from 0)."""
    shown = sorted({turn + 1 for turn in turns})
    return f' (turn{"s" if len(shown) > 1 else ""} {", ".join(str(t) for t in shown)})' if shown else ''


def why_text(cell: Cell, *, include_content: bool, fallback: str | None = None, plain: bool = False, agreement: bool = True) -> str:
    """The Why column: the judge's first finding and the turns, else the first sentence of its reasoning, with how many
    samples agreed. With content excluded only the finding kind and the turns remain.

    ``plain`` writes the reason as one plain sentence (:func:`plain_sentence`) and ``agreement=False`` leaves out the
    ``k/n judges`` tail.
    """
    detail = cell.detail
    turns = _turns_text(detail.turns)
    if not include_content:
        if not detail.kind and not detail.turns and fallback is not None:
            return fallback
        return f'{detail.kind or "violated"}{turns}'
    if plain:
        said = plain_sentence(detail.why) if detail.why else (_first_sentence(plain_sentence(detail.reasoning)) if detail.reasoning else None)
    else:
        said = detail.why or (_first_sentence(detail.reasoning) if detail.reasoning else None)
    if said is None:
        return cell.outcome.detail if fallback is None else fallback
    tail = f' · {detail.agreement[0]}/{detail.agreement[1]} judges' if agreement and detail.agreement else ''
    return f'{one_based_turns(said)}{turns}{tail}'


def class_label(classes: Sequence[NotMeasuredClass]) -> str:
    """``not measured: infra, suite``, or ``not measured`` when no trial was invalid."""
    return f'not measured: {", ".join(classes)}' if classes else 'not measured'


def deferred_text(cell: Cell) -> str:
    """``deferred to <sibling>`` for a deferred cell: the policies the backend names as owners, else ``another policy``. Raw
    tokens, unescaped, for a machine reader."""
    return f'deferred to {", ".join(cell.deferred_to) or "another policy"}'


def not_measured_reason(cell: Cell) -> str:
    """``infra: errored, provider_infra, violated in 1`` for a not-measured cell: its classes, the causes the backend named and
    the violation it still carries. Raw tokens, unescaped, for a machine reader."""
    causes = ', '.join(distinct_causes(cell))
    said = f'{", ".join(cell.outcome.classes)}: {causes}' if cell.outcome.classes and causes else ', '.join(cell.outcome.classes)
    return ', '.join(part for part in (said or _TOO_FEW_VALID, cell.outcome.detail) if part)


def _details(summary: str, body: str) -> str:
    return f'<details><summary>{summary}</summary>\n\n{body}\n\n</details>'


def _comparable(run: Run, main: Run | None) -> bool:
    """Main's run is comparable when it scored the same scenario set, eval set, env and corpus under the same schema.

    An id one of the two runs does not record cannot be compared, so a run missing the scenario set, the eval set or the env is
    not comparable. The corpus is optional: two runs that commit none are comparable, and a run with one is not comparable
    with one without. A run a signal cancelled measured only part of its suite, so it is no baseline.
    """
    if main is None or main.schema != run.schema or main.cancelled:
        return False
    required = all(kind in run.suite_ids and run.suite_ids.get(kind) == main.suite_ids.get(kind) for kind in ('scenarios', 'evals', 'env'))
    return required and run.suite_ids.get('corpus') == main.suite_ids.get('corpus')


def _gating_failed(run: Run) -> set[tuple[str, str]]:
    """The scenario × policy that fail the check: failed, gating and not ignored by ``ci.yaml``."""
    return {c.key for c in run.cells if c.outcome.outcome == 'failed' and c.outcome.level == 'gating' and c.ignored is None}


def _not_run_on_main(run: Run, main: Run) -> set[tuple[str, str]]:
    """The failed scenario × policy main has no result for: it ran another selection, so it says nothing about them."""
    ran_on_main = {c.key for c in main.cells}
    return {key for key in _gating_failed(run) if key not in ran_on_main}


def _new_vs_main(run: Run, main: Run) -> set[tuple[str, str]]:
    """The failed scenario × policy main ran and did not fail."""
    return _gating_failed(run) - _gating_failed(main) - _not_run_on_main(run, main)


def _fixed_vs_main(run: Run, main: Run) -> tuple[Cell, ...]:
    """The scenario × policy that fail the check on main and passed in this run."""
    failed_on_main = _gating_failed(main)
    return tuple(c for c in run.cells if c.key in failed_on_main and c.outcome.outcome == 'passed' and c.ignored is None)


def _days_text(options: ReportOptions) -> str | None:
    """``token expires <date> (N days)``, or ``token expired <date>``; ``None`` when no expiry is given."""
    if options.token_expires is None:
        return None
    if options.today is None:
        raise ValueError('token_expires needs today: the report core reads no clock')
    days = (options.token_expires - options.today).days
    if days < 0:
        return f'token expired {options.token_expires.isoformat()}'
    return f'token expires {options.token_expires.isoformat()} ({days} day{"s" if days != 1 else ""})'


def _expiry_note(run: Run, options: ReportOptions) -> str | None:
    if options.token_expires is None or options.today is None or (options.token_expires - options.today).days >= TOKEN_EXPIRY_NOTE_DAYS:
        return None
    return f'**Note:** {_days_text(options)}. Mint a new key with {_code(f"plurai ci token --github {run.repo or '<owner>/<repo>'}")}.'


def _scenario_count(run: Run) -> int:
    return len({c.scenario for c in run.cells} | {n.scenario for n in run.not_run})


def _size_text(run: Run) -> str:
    """``6 scenarios × 3 trials``."""
    return f'{_plural(_scenario_count(run), "scenario")} × {_plural(run.trials, "trial")}'


def _heading(ctx: _Context) -> str:
    """The verdict mark and the counts, with the size of the run at the end. It names no scenario × policy."""
    counts = ctx.counts
    parts = [f'{counts.unstable} unstable']
    if counts.not_measured or counts.verdict != 'not_measured':
        parts.append(f'{counts.not_measured} not measured')
    if counts.deferred:
        parts.append(f'{counts.deferred} deferred')
    if advisory := _advisory_text(counts):
        parts.append(f'advisory: {advisory}')
    if counts.quarantine_capped:
        parts.append(f'{_plural(counts.quarantined, "scenario")} not run')
    if regressions := ignored_regressions(ctx.run):
        parts.append(f'{_plural(len(regressions), "regression")} ignored by ci.yaml')
    tail = ' · '.join([*parts, _size_text(ctx.run)])
    if ctx.run.cancelled:
        return f'## Plurai simulation · ⛔ {CANCELLED_LEAD} · {counts.failed} failed · {tail}'
    if counts.verdict == 'failed':
        return f'## Plurai simulation · ❌ {counts.failed} failed · {tail}'
    if counts.verdict == 'not_measured':
        return f'## Plurai simulation · ⛔ {_not_measured_lead(counts)} · 0 failed · {tail}'
    return f'## Plurai simulation · ✅ 0 failed · {tail}'


def _advisory_text(counts: Tally) -> str:
    """``1 failed, 2 unstable``: the advisory cells that failed or were unstable, naming only the outcomes that have a cell."""
    return ', '.join(f'{n} {name}' for n, name in ((counts.advisory_failed, 'failed'), (counts.advisory_unstable, 'unstable')) if n)


def ignored_regressions(run: Run) -> tuple[str, ...]:
    """The regression scenarios with a scenario × policy a ``ci.yaml`` rule ignores, run or not, each once."""
    names = [c.scenario for c in run.cells if c.ignored and REGRESSION_TAG in c.tags] + [
        n.scenario for n in run.not_run if n.scenario.startswith(REGRESSION_SCENARIO_PREFIX)
    ]
    return tuple(dict.fromkeys(names))


def _not_measured_lead(counts: Tally) -> str:
    """``not measured: infra, credits (<code>, resets at …)``: every class present, with what each can say beside it."""
    credits = counts.credits
    parts = []
    for name in counts.classes:
        detail = []
        if name == 'credits' and credits is not None:
            detail = [_credits_detail(credits)]
        if name == 'suite' and counts.quarantine_capped:
            detail.append(f'too many quarantined, {counts.quarantined} of {counts.gating_scenarios} gating scenarios')
        elif name == 'suite' and counts.cells == 0:
            detail.append('everything is ignored by ci.yaml' if counts.ignored or counts.not_run else 'no scenario × policy was judged')
        parts.append(f'{name} ({", ".join(detail)})' if detail else name)
    return f'not measured: {", ".join(parts)}' if parts else 'not measured'


def _duration(seconds: int) -> str:
    return f'{round(seconds / 60)} min' if seconds >= 60 else f'{seconds} s'


def _short_id(value: str) -> str:
    """The first eight characters of an id, so the footer fits a phone; the full ids are in ``run.json``."""
    return value if len(value) <= _SHORT_ID else value[:_SHORT_ID] + '…'


def _suite_id_text(run: Run, kind: str, value: str) -> str:
    """``scenarios 7f3a1c2d…``, and ``env default (e4b1ff00…)`` when the run records the env's alias."""
    if kind == 'env' and run.env_alias:
        return f'env {_md(run.env_alias)} ({_md(_short_id(value))})'
    return _md(f'{kind} {_short_id(value)}')


def _plural(count: int, noun: str) -> str:
    return f'{count} {noun}{"" if count == 1 else "s"}'


def _footer(run: Run, options: ReportOptions, *, compared: Run | None) -> list[str]:
    ids = ', '.join(_suite_id_text(run, kind, value) for kind, value in run.suite_ids.items()) or 'not recorded'
    first = ['Suite ids: ' + ids]
    if options.stopped is None:
        selected = len({c.scenario for c in run.cells} | {n.scenario for n in run.not_run})
        of = f' of {run.suite_scenarios}' if run.suite_scenarios not in (None, selected) else ''
        count = f'{selected}{of} scenario{"" if selected == 1 and not of else "s"}'
        first.append(f'{_md(run.selection) + ", " if run.selection else ""}{count} × {_plural(run.trials, "trial")}')
        if run.threads is not None:
            first.append(_plural(run.threads, 'thread'))
        if run.duration_seconds is not None:
            first.append(_duration(run.duration_seconds))

    second: list[str] = []
    if compared is not None:
        second.append(f'Compared with main@{_md(compared.commit[:7])}' if compared.commit else 'Compared with main')
    if run.agent_models:
        models = ', '.join(_md(m) for m in run.agent_models)
        differs = compared is not None and compared.agent_models and compared.agent_models != run.agent_models
        main_ran = f' (main ran {", ".join(_md(m) for m in compared.agent_models)})' if differs and compared is not None else ''
        second.append(f'agent model {models}{main_ran}')
    if run.backend_build:
        ran = f' (main ran on {_md(compared.backend_build)})' if compared is not None and compared.backend_build not in (None, run.backend_build) else ''
        second.append(f'backend build {_md(run.backend_build)}{ran}')
    stamped = [f'action {_md(run.versions["action"])}'] if run.versions.get('action') else []
    if sdk := run.versions.get('sdk'):
        main_sdk = compared.versions.get('sdk') if compared is not None else None
        stamped.append(f'SDK {_md(sdk)}' + (f' (main ran SDK {_md(main_sdk)})' if main_sdk and main_sdk != sdk else ''))
    if options.stopped is None:
        stamped.append(f'run.json {_md(run.schema)}')
    if stamped:
        second.append(', '.join(stamped))
    if expiry := _days_text(options):
        second.append(expiry)

    lines = [' · '.join(first), *([' · '.join(second)] if second else [])]
    if options.artifacts:
        lines.append('Artifacts: ' + ' · '.join(_md(a) for a in options.artifacts))
    lines.append(f'Reproduce locally: `plurai ci run` in {_md(options.workdir)}/ at this commit')
    return lines


@dataclass(frozen=True)
class _Context:
    """What every section of one rendering reads."""

    run: Run
    options: ReportOptions
    include_content: bool
    #: Main's run when it is comparable with this one.
    main: Run | None
    #: The failed cells main ran and did not fail, when main is comparable.
    new: set[tuple[str, str]] | None
    #: The failed cells main did not run, when main is comparable.
    unseen: set[tuple[str, str]]
    #: The scenario × policy main failed and this run passed, when main is comparable.
    fixed: tuple[Cell, ...]
    counts: Tally
    #: The gating cells that failed: the cells the check fails on.
    failed: tuple[Cell, ...]

    @property
    def cells(self) -> tuple[Cell, ...]:
        """The scenario × policy that count: those a ``ci.yaml`` rule ignores are listed in their own section."""
        return tuple(c for c in self.run.cells if c.ignored is None)


def _context(run: Run, options: ReportOptions) -> _Context:
    main = options.main if _comparable(run, options.main) else None
    include_content = resolve_include_content(options.include_content, options.visibility)
    # A scenario title describes what the scenario tests, so it is run content: without content every surface names a scenario by its id.
    run = run if include_content else replace(run, scenario_titles={})
    return _Context(
        run=run,
        options=options,
        include_content=include_content,
        main=main,
        new=_new_vs_main(run, main) if main is not None else None,
        unseen=_not_run_on_main(run, main) if main is not None else set(),
        fixed=_fixed_vs_main(run, main) if main is not None else (),
        counts=tally(run),
        failed=tuple(c for c in run.cells if c.outcome.outcome == 'failed' and c.outcome.level == 'gating' and c.ignored is None),
    )


def distinct_causes(cell: Cell) -> tuple[str, ...]:
    """The causes of the cell's invalid trials, each once, in trial order."""
    return tuple(dict.fromkeys(c for c in cell.causes if c))


def _credits_detail(credits: Credits) -> str:
    """The gateway's refusal as markdown: its code, and when the allowance resets when the gateway says. Both come from the
    backend, so both are escaped."""
    return ', '.join([_md(credits.code or 'code unknown'), *([f'resets at {_md(credits.resets_at)}'] if credits.resets_at else [])])


def _why_cell(cell: Cell, ctx: _Context, why_limit: int | None, *, fallback: str | None = None, agreement: bool = True) -> str:
    said = why_text(cell, include_content=ctx.include_content, fallback=fallback, plain=True, agreement=agreement)
    return _md(said if why_limit is None else _cut(said, why_limit))


def _reasoning_details(cell: Cell, ctx: _Context) -> str | None:
    """The judge's words under a violated cell, collapsed: its finding as the judge wrote it when it differs from the plain
    sentence in the table (rubric code, evaluator expression), then its full reasoning. Judge text, so only with content."""
    detail = cell.detail
    if not ctx.include_content:
        return None
    raw = [detail.why] if detail.why and plain_sentence(detail.why) != detail.why else []
    parts = [*raw, *([detail.reasoning] if detail.reasoning else [])]
    return _details(f'{_md(policy_name(ctx.run, cell.policy))}: judge reasoning', _fenced(one_based_turns('\n\n'.join(parts)))) if parts else None


def _delta_text(ctx: _Context) -> str:
    """One line on this run against main's: the new failures, those fixed, those main did not run and those that still fail. With
    no comparable main run it says so, and why when it is known, and names how to get one."""
    main = ctx.main
    if main is None or ctx.new is None:
        given = ctx.options.main
        reason = ctx.options.main_error
        if reason is None and given is not None:
            reason = (
                "main's last run was cancelled before it finished"
                if given.cancelled
                else _main_changed(ctx.options) or 'main ran other suite ids or another schema version'
            )
        return '**vs main:** no main run to compare' + (f' ({_md(reason)})' if reason else '') + f'. To get one, {_RUN_MAIN}.'
    still = len(ctx.failed) - len(ctx.new) - len(ctx.unseen)
    parts = [
        text
        for text in (
            f'🆕 {len(ctx.new)} new failure{"" if len(ctx.new) == 1 else "s"}' if ctx.new else '',
            f'✅ {len(ctx.fixed)} fixed' if ctx.fixed else '',
            f'{len(ctx.unseen)} not run on main' if ctx.unseen else '',
            f'{still} still failing' if still else '',
        )
        if text
    ]
    label = f'main@{_md(main.commit[:7])}' if main.commit else 'main'
    return f'**vs {label}:** ' + (' · '.join(parts) or 'no failures here or on main')


def _main_changed(options: ReportOptions) -> str | None:
    """``main's suite changed since its last run`` when the ids at the PR's base commit are not the ids main's last
    run scored; ``None`` when the base ids are not known, main gave no run or its ids are the same."""
    main, base = options.main, options.base_ids
    if main is None or main.cancelled or base is None or all((base.get(kind) or None) == main.suite_ids.get(kind) for kind in _SUITE_KINDS):
        return None
    return "main's suite changed since its last run"


def _main_changed_text(ctx: _Context) -> str | None:
    """A note under the delta line when main's run is comparable and yet its suite is older than the base commit's: "vs main"
    then compares with an older suite. A run that is not comparable says the change in the delta line instead."""
    changed = _main_changed(ctx.options)
    if changed is None or ctx.main is None:
        return None
    at = f' (main@{_md(ctx.main.commit[:7])})' if ctx.main.commit else ''
    return f'**Note:** {changed}{at}, so vs main compares with an older suite. To compare with the current suite, {_RUN_MAIN}.'


def _age_text(ctx: _Context) -> str | None:
    """A note when main's run, the baseline, is more than :data:`MAIN_AGE_NOTE_DAYS` days old."""
    main, today = ctx.main, ctx.options.today
    if main is None or main.finished_on is None or today is None:
        return None
    days = (today - main.finished_on).days
    if days <= MAIN_AGE_NOTE_DAYS:
        return None
    return (
        f"**Note:** main's run is {days} days old. If the main workflow is scheduled, check that it still runs: "
        'GitHub stops scheduled workflows after 60 days without activity.'
    )


def _baseline_blocks(ctx: _Context) -> list[str]:
    """The delta line against main, and what a reader should know about the baseline it used."""
    return [_delta_text(ctx), *(note for note in (_main_changed_text(ctx), _age_text(ctx)) if note)]


#: The overrides about to expire that a rendering names; a ``ci.yaml`` with hundreds of them must not crowd out the results.
_EXPIRY_NOTES_SHOWN = 10


def _expiry_blocks(ctx: _Context) -> list[str]:
    """The notes at the top of a rendering about something that ends soon: the token, and each override about to expire (the
    first :data:`_EXPIRY_NOTES_SHOWN`, then a count of the rest)."""
    blocks = [note] if (note := _expiry_note(ctx.run, ctx.options)) else []
    soon = [n for n in ctx.run.override_notes if n.kind == 'expires_soon']
    blocks += [f'**Note:** {override_note_text(n)}.' for n in soon[:_EXPIRY_NOTES_SHOWN]]
    if len(soon) > _EXPIRY_NOTES_SHOWN:
        blocks.append(f'**Note:** {len(soon) - _EXPIRY_NOTES_SHOWN} more overrides expire soon; the dates are in `ci.yaml`.')
    return blocks


def _row(cells: Sequence[str]) -> str:
    """A table row; an empty cell stays one space wide."""
    return '|' + '|'.join(f' {cell} ' if cell else ' ' for cell in cells) + '|'


def _status_counts(cells: Sequence[Cell]) -> tuple[int, ...]:
    """How many of ``cells`` show each status, in :data:`_STATUSES` order."""
    icons = [status_icon(c) for c in cells]
    return tuple(icons.count(icon) for icon, _, _ in _STATUSES)


def _worst(counts: Sequence[int]) -> str:
    present = {icon for (icon, _, _), n in zip(_STATUSES, counts, strict=True) if n}
    return next((icon for icon in _STATUS_RANK if icon in present), '➖')


@dataclass(frozen=True)
class _Scenario:
    """One scenario of the run with its counted scenario × policy."""

    id: str
    cells: tuple[Cell, ...]

    @property
    def counts(self) -> tuple[int, ...]:
        """Each status on the basis the heading counts, so the totals row and the heading agree: failed, unstable and not
        measured over the policies that can fail the check; passed over every policy, since the heading has no passed count."""
        return _status_counts([c for c in self.cells if c.outcome.level == 'gating' or c.outcome.outcome == 'passed'])

    @property
    def advisory(self) -> tuple[int, int]:
        """The advisory policies that failed and that were unstable: they never fail the check, and the heading names them apart."""
        outcomes = [c.outcome.outcome for c in self.cells if c.outcome.level == 'advisory']
        return outcomes.count('failed'), outcomes.count('unstable')

    @property
    def icon(self) -> str:
        """The worst of every policy, advisory ones too, as the policy rows under the scenario show them."""
        return _worst(_status_counts(self.cells))

    @property
    def needs_attention(self) -> bool:
        return self.icon in ('❌', '⚠️')


def _catastrophic_violation(scenario: _Scenario) -> bool:
    return any('catastrophic' in c.tags and c.outcome.violated for c in scenario.cells)


def _scenarios(ctx: _Context) -> list[_Scenario]:
    """The scenarios with a counted scenario × policy, worst first; within one icon, those with a violated catastrophic
    policy lead, then run order. Policies are ordered by icon inside a scenario."""
    grouped: dict[str, list[Cell]] = {}
    for cell in ctx.cells:
        grouped.setdefault(cell.scenario, []).append(cell)
    scenarios = [_Scenario(sid, tuple(sorted(cells, key=lambda c: _STATUS_RANK.index(status_icon(c))))) for sid, cells in grouped.items()]
    return sorted(scenarios, key=lambda s: (_STATUS_RANK.index(s.icon), not _catastrophic_violation(s)))


def _overview_block(ctx: _Context, shown: Sequence[_Scenario], everything: Sequence[_Scenario]) -> list[str]:
    """The table the results open with: a row per scenario with its icon, its name and the count of each status as the heading
    counts it (see :attr:`_Scenario.counts`), then a totals row over every scenario of the run, shown or not, which matches
    the heading. When an advisory policy failed or was unstable, an Advisory column counts those apart."""
    if not everything:
        return []
    advisory = bool(ctx.counts.advisory_failed or ctx.counts.advisory_unstable)
    columns = [column for _, _, column in _COUNTED] + (['Advisory'] if advisory else [])

    def cells(counts: Sequence[int], extra: tuple[int, int]) -> list[str]:
        return [str(n) for n in counts[: len(_COUNTED)]] + ([_advisory_marks(*extra)] if advisory else [])

    rows = [_row([s.icon, _labelled(scenario_name(ctx.run, s.id), s.id), *cells(s.counts, s.advisory)]) for s in shown]
    totals = [sum(s.counts[i] for s in everything) for i in range(len(_COUNTED))]
    total_advisory = (sum(s.advisory[0] for s in everything), sum(s.advisory[1] for s in everything))
    rows.append(_row(['', f'**Total: {_plural(len(everything), "scenario")}**', *(f'**{text}**' if text else '' for text in cells(totals, total_advisory))]))
    lines = ['### Scenarios', _row(['', 'Scenario', *columns]), _row(['---'] * (2 + len(columns))).replace(' --- ', '---'), *rows]
    if advisory:
        lines.append('\nAdvisory: policies that failed or were unstable and never fail the check; the other columns leave them out.')
    return ['\n'.join(lines)]


def _advisory_marks(failed: int, unstable: int) -> str:
    """``1 ❌ · 2 ⚠️``: the advisory policies that failed and were unstable, naming only the outcomes that have one."""
    return ' · '.join(f'{n} {icon}' for n, icon in ((failed, '❌'), (unstable, '⚠️')) if n)


def _summary_counts(counts: Sequence[int], advisory: tuple[int, int] = (0, 0)) -> str:
    """``5 failed · 2 unstable · 6 passed · advisory: 1 unstable``: each counted status the scenario has, then its advisory
    failures as the heading words them; ``N/A`` for a scenario that has no other."""
    said = [f'{n} {words}' for (_, words, _), n in zip(_COUNTED, counts, strict=False) if n]
    if extra := ', '.join(f'{n} {name}' for n, name in zip(advisory, ('failed', 'unstable'), strict=True) if n):
        said.append(f'advisory: {extra}')
    return ' · '.join(said) or 'N/A'


def _tree_why(cell: Cell, ctx: _Context) -> str:
    """What the tree says beside a scenario × policy: the plain reason, turns and judges of a violated one, the causes of one
    not measured, the owner of a deferred one."""
    outcome = cell.outcome.outcome
    if outcome in ('failed', 'unstable'):
        new = '🆕 ' if ctx.new is not None and cell.key in ctx.new else 'not run on main · ' if cell.key in ctx.unseen else ''
        said = new + _why_cell(cell, ctx, None)
    elif outcome == 'not_measured':
        causes = _main_cause_words(cell)
        said = f'{class_label(cell.outcome.classes)} — {causes}' if causes else _TOO_FEW_VALID
        said += f', {cell.outcome.detail}' if cell.outcome.detail else ''
    elif outcome == 'deferred':
        said = _md(deferred_text(cell))
    else:
        said = ''
    return ' · '.join(part for part in (said, 'retried once' if cell.retried else '') if part)


def _folded(ctx: _Context, icon: str, label: str, cells: Sequence[Cell]) -> str:
    """One closed line, ``✅ Passed: 6 policies``, that opens to the policies it counts. A deferred policy says whose it is."""
    items = []
    for cell in cells:
        why = _md(deferred_text(cell)) if cell.outcome.outcome == 'deferred' else ''
        items.append(f'- {_md(policy_name(ctx.run, cell.policy))} {_code(_short_policy_id(cell.policy))}' + (f' — {why}' if why else ''))
    return _details(f'{icon} {label}: {_plural(len(cells), "policy").replace("policys", "policies")}', '\n'.join(items))


def _scenario_tree(ctx: _Context, scenario: _Scenario) -> str:
    """One scenario as a ``<details>`` block: open when a scenario × policy failed. Inside, its id, a narrow table with a row
    per policy that failed, was unstable or not measured (an icon, a mark per trial, its tier), the reason of each under the
    table, and the judge reasoning of each violated policy collapsed.

    The passed policies are folded into one line when the scenario has a failure, else they are rows too. The N/A policies are
    always folded."""
    name = scenario_name(ctx.run, scenario.id)
    everything = _status_counts(scenario.cells)
    opened = ' open' if everything[0] else ''
    failing = everything[0] + everything[1] > 0
    passed = [c for c in scenario.cells if status_icon(c) == '✅']
    not_applicable = [c for c in scenario.cells if status_icon(c) == '➖']
    shown = [c for c in scenario.cells if c not in not_applicable and not (failing and c in passed)]
    blocks = [_code(scenario.id)]
    if scenario.id.startswith(REGRESSION_SCENARIO_PREFIX):
        blocks.append(_regression_line(ctx.run.fail_on))
    if shown:
        rows = [
            _row(
                [
                    status_icon(c),
                    _labelled(policy_name(ctx.run, c.policy), _short_policy_id(c.policy)),
                    ''.join(TRIAL_MARKS[v] for v in c.verdicts),
                    tier_text(c),
                ]
            )
            for c in shown
        ]
        blocks.append('\n'.join([_row(['', 'Policy', 'Trials', 'Severity']), '|---|---|---|---|', *rows]))
        reasons = [f'- {status_icon(c)} **{_md(policy_name(ctx.run, c.policy))}** — {why}' for c in shown if (why := _tree_why(c, ctx))]
        if reasons:
            blocks.append('\n'.join(reasons))
    blocks += [block for c in scenario.cells if c.outcome.violated and (block := _reasoning_details(c, ctx))]
    if failing and passed:
        blocks.append(_folded(ctx, '✅', 'Passed', passed))
    if not_applicable:
        blocks.append(_folded(ctx, '➖', 'Not applicable', not_applicable))
    body = '\n\n'.join(blocks)
    return f'<details{opened}><summary>{scenario.icon} {_md(name)} — {_summary_counts(scenario.counts, scenario.advisory)}</summary>\n\n{body}\n\n</details>'


def _results_blocks(ctx: _Context, *, attention_only: bool = False) -> list[str]:
    """The overview, the scenarios fixed against main, and the test tree. ``attention_only`` keeps the failed and unstable
    scenarios in the overview and the tree, for a text field too small for all of them, and says where the others are."""
    everything = _scenarios(ctx)
    shown = [s for s in everything if s.needs_attention] if attention_only else everything
    blocks = _overview_block(ctx, shown, everything)
    if ctx.fixed:
        blocks.append(f'### Fixed ({len(ctx.fixed)})\n' + '\n'.join(f'- ✅ {_name(ctx.run, c)}' for c in ctx.fixed))
    if everything:
        blocks.append(f'### Results by scenario\n{_MARKS_LEGEND} {_fail_on_text(ctx.run.fail_on)}')
        blocks += [_scenario_tree(ctx, s) for s in shown]
    if hidden := len(everything) - len(shown):
        blocks.append(f'The other {_plural(hidden, "scenario")} are in the {_job_summary(ctx)}.')
    if not ctx.include_content and any(c.outcome.violated for c in ctx.cells):
        blocks.append(_WITHHELD)
    return blocks


def _notes(ctx: _Context) -> list[str]:
    run = ctx.run
    notes = [scrub_surrogates(' '.join(n.split())) for n in ctx.options.notes]
    unspecified = sorted({c.policy for c in ctx.cells if not any(tag in TIERS for tag in c.tags)})
    unless = ' unless they are regressions' if REGRESSION_TAG in run.fail_on else ''
    notes += [f'policy {_md(policy)} has no tier: its scenario × policy are treated as advisory{unless}' for policy in unspecified]
    if run.quarantined:
        notes.append(f'{_plural(len(run.quarantined), "scenario")} quarantined as unsolvable, not run: {", ".join(_md(q) for q in run.quarantined)}')
        if ctx.counts.quarantine_capped:
            notes.append(f'more than {QUARANTINE_CAP_PERCENT}% of the selected gating scenarios are quarantined, so the run is not measured: suite')
    if any(NEEDS_REDEPLOY in c.causes for c in ctx.cells):
        notes.append('the mock environment needs a redeploy and is not retried: run `/use-env`')
    notes += [f'cause {_md(cause)} is not one this report knows; its trials are counted as infra' for cause in run.unknown_causes]
    notes += [f'tier {_md(tier)} is not one this report knows; its scenario × policy are gating and not measured: infra' for tier in run.unknown_tiers]
    notes += _crash_notes(ctx.cells)
    notes += [override_note_text(n) for n in run.override_notes if n.kind != 'expires_soon']
    if new := newly_quarantined(run, ctx.main):
        notes.append(f"newly quarantined since main's last run: {', '.join(_md(q) for q in new)}")
    return notes


def _crash_notes(cells: Sequence[Cell]) -> list[str]:
    """One note per scenario whose agent raised in some trial: a cell that still passes on its other trials must not hide it."""
    crashed: dict[str, set[int]] = {}
    for cell in cells:
        for trial, cause in enumerate(cell.causes, start=1):
            if cause == 'agent_raised':
                crashed.setdefault(cell.scenario, set()).add(trial)
    return [
        f'the agent raised an error in {_plural(len(trials), "trial")} of {_code(scenario)} ({numbers}); those trials are not measured'
        for scenario, trials in crashed.items()
        if (numbers := ', '.join(str(t) for t in sorted(trials)))
    ]


def _is_quiet(note: str) -> bool:
    """Whether no reader acts on ``note``: it goes in "Run details" instead of the visible notes."""
    return note.startswith(_QUIET_NOTES)


def override_note_text(note: OverrideNote) -> str:
    """The line a report shows for a remark about a ``ci.yaml`` rule."""
    rule = _md(note.rule)
    if note.kind == 'expired':
        until = f' on {note.expires.isoformat()}' if note.expires else ''
        return f'override expired: {rule} ended{until} and no longer applies'
    if note.kind == 'expires_soon':
        left = note.days_left or 0
        when = 'today' if left == 0 else f'in {_plural(left, "day")}'
        until = f' ({note.expires.isoformat()})' if note.expires else ''
        return f'override for {rule} expires {when}{until}'
    if note.kind == 'matches_nothing':
        return f'override {rule} matches nothing in the suite'
    if note.kind == 'now_passing':
        return f'override {rule}: now passing, remove the rule'
    return f'{rule} is on the smoke list and every scenario × policy of it is ignored, so it is not run'


def _notes_block(notes: Sequence[str]) -> list[str]:
    """The notes a reader can act on; the quiet ones go in "Run details"."""
    shown = [n for n in notes if not _is_quiet(n)]
    return ['### Notes\n' + '\n'.join(f'- {n}' for n in shown)] if shown else []


def _run_details(footer: Sequence[str], notes: Sequence[str]) -> list[str]:
    """The closed "Run details" section: the suite ids, the versions, the artifacts, how to reproduce, and the quiet notes."""
    lines = [*footer, *(n for n in notes if _is_quiet(n))]
    return [_details('Run details', '\n'.join(f'- {line}' for line in lines))]


def _cancelled_blocks(run: Run) -> list[str]:
    """The paragraph of a cancelled run: that the results are those measured before the cancel."""
    if not run.cancelled:
        return []
    return [f'{CANCELLED_FIX} The results below are what was measured before the cancel; every other scenario × policy is not measured.']


def _detail_blocks(ctx: _Context) -> list[str]:
    """The sections after the test tree: what ``ci.yaml`` ignores, what was deferred, what could not be measured, the notes."""
    return [*_ignored_blocks(ctx), *_deferred_blocks(ctx), _not_measured_block(ctx), *_notes_block(_notes(ctx))]


def _expiry_text(ignored: Ignored) -> str:
    return f' · expires {ignored.expires.isoformat()}' if ignored.expires else ''


def _reason_text(ignored: Ignored) -> str:
    return f'"{_md(ignored.reason)}"{_expiry_text(ignored)}'


def _ignored_outcome(cell: Cell) -> str:
    """What an ignored scenario × policy concluded: ``failed 3/3``, ``unstable 1/3``, ``passed``, ``not measured: infra``."""
    result = cell.outcome
    label = class_label(result.classes) if result.outcome == 'not_measured' else result.outcome.replace('_', ' ')
    return f'{label} {result.ratio}' if result.outcome in ('failed', 'unstable') else label


def _ignored_blocks(ctx: _Context) -> list[str]:
    """Ignored by ci.yaml: each ignored scenario × policy with the outcome it computed, then the scenarios that were not run,
    grouped by the rule that ignores them. Nothing here counts."""
    lines = [f'- {_name(ctx.run, c)}: {_reason_text(c.ignored)} · {_ignored_outcome(c)}' for c in ctx.run.cells if c.ignored is not None]
    groups: dict[tuple[Ignored, ...], list[NotRun]] = {}
    for skipped in ctx.run.not_run:
        groups.setdefault(skipped.ignored, []).append(skipped)
    for rules, skipped in groups.items():
        named = ', '.join(_md(rule.rule) for rule in rules)
        said = '; '.join(_reason_text(rule) for rule in rules)
        lines.append(f'- {named} ({_plural(len(skipped), "scenario")}, not run): {said}')
    return ['### Ignored by ci.yaml (does not count)\n' + '\n'.join(lines)] if lines else []


def _deferred_blocks(ctx: _Context) -> list[str]:
    """Deferred (does not count): each counted scenario × policy the backend deferred to a sibling policy, naming the sibling."""
    lines = [f'- {_name(ctx.run, c)}: {_md(deferred_text(c))}' for c in ctx.cells if c.outcome.outcome == 'deferred']
    return ['### Deferred (does not count)\n' + '\n'.join(lines)] if lines else []


def _main_cause_words(cell: Cell) -> str:
    """The cause of the most invalid trials (the first in trial order on a tie) in plain words, its raw token beside it and the
    raw tokens of the cell's other causes after it, so a line says one cause and a search still finds each; empty for none."""
    causes = [c for c in cell.causes if c]
    if not causes:
        return ''
    main = max(dict.fromkeys(causes), key=causes.count)
    words = _cause_words(main, cell)
    others = [_md(c) for c in dict.fromkeys(causes) if c != main]
    return f'{words}; also {", ".join(others)}' if others else words


def _cause_words(cause: str, cell: Cell) -> str:
    """A cause in plain words with its raw token beside it for a search; a credits cause carries the gateway's code and reset."""
    if cause == 'credits' and cell.credits is not None:
        return f'{CAUSES[cause].words} ({_credits_detail(cell.credits)})'
    words = ' and '.join(part.words for part in cause_parts(cause))
    return f'{words} ({_md(cause)})'


def _next_steps(cells: Sequence[Cell], *, capped: bool) -> str:
    """One sentence per class present: what to do about each cause in it. A joined cause takes the steps of the parts that
    earn its class."""
    steps: dict[NotMeasuredClass, list[str]] = {}
    for cell in cells:
        for cause in distinct_causes(cell):
            earned = INVALID_CLASS[classify(cause)[0]]
            for entry in cause_parts(cause):
                if entry.cls == earned and entry.next_step not in steps.setdefault(earned, []):
                    steps[earned].append(entry.next_step)
    if capped:
        steps.setdefault('suite', []).append(CAUSES[QUARANTINED].next_step)
    return ' '.join(f'{name}: {"; ".join(steps[name])}.' for name in CLASS_ORDER if name in steps)


def _not_measured_block(ctx: _Context) -> str:
    """The Not measured section: what to do about each class present, then each scenario × policy that could not be measured
    and, when quarantine caps the run, each scenario that was not run."""
    cells = [c for c in ctx.cells if c.outcome.outcome == 'not_measured']
    lines = []
    for c in cells:
        causes = _main_cause_words(c)
        said = f'{", ".join(c.outcome.classes)} — {causes}' if c.outcome.classes and causes else _TOO_FEW_VALID
        detail = f', {c.outcome.detail}' if c.outcome.detail else ''
        advisory = ' (advisory, not counted)' if c.outcome.level == 'advisory' else ''
        retried = ' (retried once)' if c.retried else ''
        lines.append(f'- {_name(ctx.run, c)}: {said}{detail}{retried}{advisory}')
    capped = ctx.counts.quarantine_capped
    if capped:
        lines += [f'- {_md(q)}: suite — {CAUSES[QUARANTINED].words}, not run' for q in ctx.run.quarantined]
    steps = _next_steps(cells, capped=capped)
    measured_again = _retried_and_measured(ctx)
    if measured_again:
        note = f'{len(measured_again)} scenario × policy retried once for an infra cause and then measured'
        lines.append(f'- {note}: {", ".join(measured_again)}' if lines else f'- none ({note})')
    return '\n'.join(['### Not measured', *([steps] if steps else []), *(lines or ['- none'])])


def _retried_and_measured(ctx: _Context) -> list[str]:
    """The counted scenario × policy that took a second pass and were then measured, named."""
    return [_name(ctx.run, c) for c in ctx.cells if c.retried and c.outcome.outcome != 'not_measured']


def _join(heading: str, blocks: Sequence[str], footer: Sequence[str]) -> str:
    return '\n\n'.join([heading, *blocks]) + '\n\n' + '\n'.join(footer) + '\n'


#: Markup a heading cannot carry: the check's title shows it as typed, so it is dropped rather than escaped.
_HEADING_MARKUP = re.compile(r'[\\`*\[\]<>|]')
#: A stop's words in the heading are cut to this many characters; the paragraph under it has the rest.
_STOP_HEADING_LIMIT = 100


def stopped_heading(stopped: Stopped) -> str:
    """The heading line of a run that ended at a stop: the verdict mark and the stop's words, plain and short."""
    words = ' '.join(_HEADING_MARKUP.sub('', scrub_surrogates(stopped.reason)).split())
    if len(words) > _STOP_HEADING_LIMIT:
        words = words[: _STOP_HEADING_LIMIT - 1].rsplit(' ', 1)[0] + '…'
    return f'## Plurai simulation · ⛔ {words}'


def _report_parts(run: Run, options: ReportOptions | None, *, attention_only: bool = False) -> tuple[str, list[str], list[str]]:
    """The report's heading, the blocks after it and its footer: the closed "Run details" section. ``attention_only`` keeps
    only the failed and unstable scenarios in the overview and the tree."""
    ctx = _context(run, options or ReportOptions())
    if (stopped := ctx.options.stopped) is not None:
        notes = [scrub_surrogates(' '.join(n.split())) for n in ctx.options.notes]
        return stopped_heading(stopped), [stopped.fix, *_notes_block(notes)], _run_details(_footer(run, ctx.options, compared=None), notes)
    blocks = [
        *_cancelled_blocks(ctx.run),
        *_expiry_blocks(ctx),
        *_baseline_blocks(ctx),
        *_results_blocks(ctx, attention_only=attention_only),
        *_detail_blocks(ctx),
    ]
    return _heading(ctx), blocks, _run_details(_footer(run, ctx.options, compared=ctx.main), _notes(ctx))


def render_heading(run: Run, options: ReportOptions | None = None) -> str:
    """The heading line of the report: the verdict, the bare counts and the size of the run. A stopped run, given in
    ``options.stopped``, has the stop's heading."""
    return _report_parts(run, options)[0]


def render_report(run: Run, options: ReportOptions | None = None, *, suite_diff: str | None = None, limit_chars: int = CHECK_RUN_LIMIT_CHARS) -> str:
    """``report.md``, which is also the check run's text: the verdict heading, the delta against main, the scenario overview,
    the test tree, the other sections and the closed "Run details". ``suite_diff``, the rendered "Suite changed vs base"
    block, goes after the results and before "Run details".

    The text stays within ``limit_chars`` (UTF-16 units). When the whole tree does not fit, the overview and the tree keep
    the failed and unstable scenarios and a line points to the job summary for the others; if that is still over, sections
    are cut as :func:`render_summary` cuts them.
    """
    diff = [suite_diff.strip('\n')] if suite_diff and suite_diff.strip() else []
    for attention_only in (False, True):
        heading, blocks, footer = _report_parts(run, options, attention_only=attention_only)
        if _utf16_length(text := _join(heading, [*blocks, *diff], footer)) <= limit_chars:
            return text
    return _fit(
        heading, [*blocks, *diff], footer, limit_chars, size=_utf16_length, pointer='cut to fit the check run limit; the full results are in the job summary'
    )


def _utf16_length(text: str) -> int:
    """``text`` in UTF-16 code units, the strictest reading of a character count."""
    return len(text.encode('utf-16-le')) // 2


def is_plain_web_address(url: str) -> bool:
    """Whether ``url`` is an ``http(s)`` address with no space, markdown syntax or HTML in it, so it is safe as a link target."""
    return _WEB_ADDRESS.fullmatch(url) is not None


def _job_summary(ctx: _Context) -> str:
    """The words "job summary", a link to the run's page when ``run_url`` is given."""
    if (url := ctx.options.run_url) is None:
        return 'job summary'
    if not is_plain_web_address(url):
        raise ValueError(f'run_url is not a plain http(s) address: {url!r}')
    return f'[job summary]({url})'


def _pointer_to_results(ctx: _Context, counted: Sequence[str] = ()) -> str:
    """Where the results are, in one sentence with one link: the job summary (linked to the run's page when ``run_url`` is
    given) has every result, including the ``counted`` ones the comment does not list, and the artifacts. No full stop."""
    sentence = f'The {_job_summary(ctx)} has every result'
    if counted:
        sentence += f', including the {_and_list(counted)}'
    if artifacts := ' · '.join(_md(a) for a in ctx.options.artifacts):
        sentence += f'{"," if counted else ""} and the artifacts {artifacts}'
    return sentence


def _pointer_to_details(ctx: _Context) -> str:
    """Where a stopped run's details are, in one sentence with one link: the job summary (linked as in
    :func:`_pointer_to_results`) and the artifacts. A stop has no results to point to. No full stop."""
    sentence = f'The {_job_summary(ctx)} has the run details'
    artifacts = [_md(a) for a in ctx.options.artifacts]
    if artifacts:
        sentence += f' and the {"artifact" if len(artifacts) == 1 else "artifacts"} {" · ".join(artifacts)}'
    return sentence


def render_comment(run: Run, options: ReportOptions | None = None, *, limit_chars: int = COMMENT_LIMIT_CHARS - SUITE_DIFF_COMMENT_CHARS - _APPEND_SLACK) -> str:
    """The PR comment body: the counts heading, the delta against main, up to five scenario × policy and a pointer to the full
    results in the job summary.

    The five are the new failures when main is comparable and any failure is new, else the failures. Each is one line with a
    plain reason cut at a word. The comment opens with :data:`COMMENT_MARKER`, so a poster finds it again and edits it in
    place. Every count in the heading is named in the pointer sentence.

    The body stays within ``limit_chars`` (UTF-16 units): it lists fewer rows until it fits and says how many it did not list.
    The default leaves :data:`SUITE_DIFF_COMMENT_CHARS` of GitHub's limit for the suite diff the action appends. The heading
    and the pointer always stay, so a limit below their size cannot be met.
    """
    ctx = _context(run, options or ReportOptions())
    if (stopped := ctx.options.stopped) is not None:
        return f'{COMMENT_MARKER}\n{stopped_heading(stopped)}\n\n{stopped.fix}\n\n{_pointer_to_details(ctx)}.\n'
    counts = ctx.counts
    new = [c for c in ctx.failed if ctx.new is not None and c.key in ctx.new]
    candidates = new or list(ctx.failed)
    label = 'New failures' if new else 'Failed' if ctx.new is None or ctx.unseen else 'Still failing'
    others = [
        f'{n} {name}'
        for n, name in (
            (counts.unstable, 'unstable'),
            (counts.not_measured, 'not measured'),
            (counts.deferred, 'deferred'),
            (counts.advisory_failed, 'advisory failed'),
            (counts.advisory_unstable, 'advisory unstable'),
        )
        if n
    ]
    where = _pointer_to_results(ctx, others)
    summary = _job_summary(ctx)
    pointer = ''
    if counts.ignored or run.not_run:
        parts = ([f'{counts.ignored} scenario × policy'] if counts.ignored else []) + (
            [f'{_plural(len(run.not_run), "scenario")} not run'] if run.not_run else []
        )
        pointer += f'Ignored by ci.yaml, listed in the {summary}: {", ".join(parts)}.\n'
    if notes := [n for n in _notes(ctx) if not _is_quiet(n)]:
        pointer += f'The {summary} has {_plural(len(notes), "note")}.\n'

    def compose(keep: int) -> str:
        shown = candidates[:keep]
        blocks = _cancelled_blocks(run)
        blocks += _expiry_blocks(ctx)
        blocks += _baseline_blocks(ctx)
        if shown:
            rows = [f'- {status_icon(c)} **{_name(ctx.run, c)}** — {_why_cell(c, ctx, _COMMENT_WHY_LIMIT, agreement=False)}' for c in shown]
            blocks.append('\n'.join([f'**{label}**', *rows]))
        if len(shown) < len(candidates):
            rest = len(candidates) - len(shown)
            blocks.append(f'… and {rest} more in the {summary}.' if shown else f'The {summary} lists all {len(candidates)}.')
        if shown and not ctx.include_content:
            blocks.append(_WITHHELD)
        return f'{COMMENT_MARKER}\n' + '\n\n'.join([_heading(ctx), *blocks]) + f'\n\n{where}.\n{pointer}'

    for keep in range(min(len(candidates), _COMMENT_ROWS), 0, -1):
        if _utf16_length(body := compose(keep)) <= limit_chars:
            return body
    return compose(0)


def render_summary(run: Run, options: ReportOptions | None = None, *, limit_bytes: int = SUMMARY_LIMIT_BYTES - SUITE_DIFF_SUMMARY_BYTES - _APPEND_SLACK) -> str:
    """The job summary body: the whole report (every scenario in the tree), cut to ``limit_bytes``. The default leaves
    :data:`SUITE_DIFF_SUMMARY_BYTES` of GitHub's limit for the suite diff the action appends."""
    heading, blocks, footer = _report_parts(run, options)
    return _fit(heading, blocks, footer, limit_bytes, size=lambda text: len(text.encode()), pointer=_CUT_POINTER)


def _fit(heading: str, blocks: list[str], footer: Sequence[str], limit: int, *, size: Callable[[str], int], pointer: str) -> str:
    """The report within ``limit`` as ``size`` measures it.

    A heading, a table and a section are never cut in the middle. ``<details>`` blocks go first, last block first, each
    replaced by a line that says ``pointer``. If the text is still over the limit, whole sections go from the end and one
    line says ``pointer``. The heading and the footer always stay, so a limit below their size cannot be met.
    """
    blocks = list(blocks)
    for index in range(len(blocks) - 1, -1, -1):
        if size(_join(heading, blocks, footer)) <= limit:
            return _join(heading, blocks, footer)
        if blocks[index].startswith('<details'):
            summary = _SUMMARY_TAG.search(blocks[index])
            blocks[index] = f'> {summary.group(1) if summary else "details"}: {pointer}.'
    if size(_join(heading, blocks, footer)) <= limit:
        return _join(heading, blocks, footer)
    last = f'> Sections after the heading are {pointer}.'
    while blocks and size(_join(heading, [*blocks, last], footer)) > limit:
        blocks.pop()
    return _join(heading, [*blocks, last], footer)
