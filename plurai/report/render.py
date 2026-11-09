"""The deterministic markdown renderings of a run: ``report.md``, the job summary body and the PR comment body.

The same input renders the same bytes: no clock, no network, no model, no environment. What the report cannot know it
takes as plain input in :class:`ReportOptions`: main's run, the repository's visibility, notes, the token's expiry,
today's date and, for a run that ended before any thread, the :class:`Stopped` heading and fix. The text a reader sees
says "scenario × policy" and "suite ids"; "cell" and "pin" appear only in identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from plurai.report.outcome import CLASS_ORDER, INVALID_CLASS, NotMeasuredClass
from plurai.report.run import CAUSES, NEEDS_REDEPLOY, QUARANTINED, REPO_SHAPE, TIERS, UNKNOWN_CAUSE, Cell, Credits, Run, classify, scrub_surrogates
from plurai.report.tally import QUARANTINE_CAP_PERCENT, Tally, tally

#: GitHub caps a job summary at 1 MiB.
SUMMARY_LIMIT_BYTES = 1024 * 1024

#: GitHub rejects a comment body over this many characters.
COMMENT_LIMIT_CHARS = 65_536

#: The sticky PR comment is found again by this marker and edited in place.
COMMENT_MARKER = '<!-- plurai-simulation-report -->'

#: Under this many days to expiry the token line is also a note at the top of the report.
TOKEN_EXPIRY_NOTE_DAYS = 14

#: What a scenario × policy with no tier shows where its tier would be.
NO_TIER = 'no tier'

Visibility = Literal['public', 'private', 'internal']

_REASONING_LIMIT = 140

_SHORT_ID = 8
_TOO_FEW_VALID = 'fewer than 2 valid trials'
_WITHHELD = 'Judge text is withheld; set `include_content: true` in the workflow to show it.'
_WEB_ADDRESS = re.compile(r'https?://[^\s<>`\\\[\]*|]+')
#: How many failed scenario × policy the heading names before "and K more", so the heading stays short.
_HEADING_NAMES = 5
_SELF_CONSISTENCY = re.compile(r'^\[self-consistency[^\]]*\]\s*')
_SUMMARY_TAG = re.compile(r'<summary>(.*?)</summary>', re.DOTALL)
#: A comment over its limit first cuts each judge reason to this many characters, then drops rows from the end.
_COMMENT_WHY_LIMIT = 300
_CUT_POINTER = 'cut to fit the job summary limit; the full list is in report.md'
_CUT_POINTER_COMMENT = 'see the report artifact (report.md) for the full list'


@dataclass(frozen=True)
class Stopped:
    """A run that ended before any thread: the heading it earns and the one paragraph that says what to do.

    Build one with a constructor below; the renderers then show the heading, the paragraph and the footer, and no counts or
    tables. ``reason`` and ``fix`` are markdown the constructors have already escaped.
    """

    #: What follows the verdict mark in the heading, such as ``not measured: no secrets (fork PR)``.
    reason: str
    fix: str

    @classmethod
    def no_secrets(cls, kind: Literal['fork PR', 'Dependabot']) -> Stopped:
        """The job has no ``PLURAI_TOKEN``: a pull request from a fork, or one Dependabot opened."""
        if kind == 'fork PR':
            fix = (
                'Nothing ran: the workflow has no `PLURAI_TOKEN` on a pull request from a fork. '
                'A maintainer can push the branch to this repository to measure it.'
            )
        else:
            fix = (
                "Nothing ran: Dependabot's secrets do not include `PLURAI_TOKEN`. "
                'Give Dependabot the secret with `gh secret set --app dependabot PLURAI_TOKEN`.'
            )
        return cls(f'not measured: no secrets ({_md(kind)})', fix)

    @classmethod
    def refused(cls, detail: str) -> Stopped:
        """The run declined to start on what it checked first; ``detail`` names the problem and the value to change."""
        return cls(f'refused — {_md(detail)}', 'Nothing ran. Change what the heading names and run again.')

    @classmethod
    def token_rejected(cls, repo: str | None = None) -> Stopped:
        """The backend answered 401: the token is expired, revoked or wrong."""
        if repo and not REPO_SHAPE.fullmatch(repo):
            raise ValueError(f'repo is not an owner/name: {repo!r}')
        mint = _code(f'plurai ci token --github {repo or "<owner>/<repo>"}')
        return cls('token rejected', f'Nothing ran: the backend rejected `PLURAI_TOKEN`, which is expired, revoked or wrong. Mint a new key with {mint}.')

    @classmethod
    def token_refused(cls, route: str) -> Stopped:
        """The backend answered 403 on ``route``: the token is valid but may not do this."""
        return cls(
            f'token refused for {_md(route)}',
            'Nothing ran: the backend refused `PLURAI_TOKEN` for that route. Use a key minted for this repository with `plurai ci token`.',
        )

    @classmethod
    def not_measured(cls, name: NotMeasuredClass, why: str, fix: str) -> Stopped:
        """A step before the first thread failed for an infra, suite or credits cause. ``why`` is the plain-words cause and
        ``fix`` the remedy that cause calls for, a sentence ending in a full stop: the class alone does not decide it (a
        committed id the backend does not have is not fixed by running again)."""
        return cls(f'not measured: {name} ({_md(why)})', f'Nothing ran. {fix[0].upper()}{fix[1:]}')


@dataclass(frozen=True)
class ReportOptions:
    """What the renderers take besides the run."""

    #: Main's last run, for the "vs main" column. Used only when it scored the same four suite ids under the same schema.
    main: Run | None = None
    #: ``include_content``: ``True`` or ``False`` decides; ``None`` is auto, on only for a private repository.
    include_content: bool | None = None
    visibility: Visibility | None = None
    #: Why ``main`` is not given although the caller looked for it (an unreadable file, another schema), for the note. Words
    #: only: the runner's file paths are not for the reader.
    main_error: str | None = None
    #: Config and infra remarks the caller knows. They are listed and never counted. Markdown the caller composed: the
    #: renderer keeps it as given, so text from the backend goes in escaped.
    notes: Sequence[str] = ()
    artifacts: Sequence[str] = ()
    #: Where the whole run is shown (the job's page), for the PR comment to link. A plain ``http(s)`` address.
    run_url: str | None = None
    #: When the CI token expires, and today's date to count the days from. Both or neither.
    token_expires: date | None = None
    today: date | None = None
    #: Set when the run ended before any thread: every rendering then shows this instead of counts and tables.
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
_BACKTICK_RUN = re.compile('`+')


def _md(text: str) -> str:
    """``text`` as one line of inline markdown that shows as written: lone surrogates replaced, whitespace and newlines
    folded, and every character that starts markup, a table column or HTML escaped (the backslash first)."""
    folded = ' '.join(scrub_surrogates(text).split())
    return _MARKDOWN_SPECIAL.sub(r'\\\1', folded.replace('&', '&amp;'))


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


def _name(cell: Cell) -> str:
    return _md(f'{cell.scenario} / {cell.policy}')


def _tier(cell: Cell) -> str:
    return _md(' · '.join(cell.tags) or NO_TIER)


def _policy_tier(cell: Cell) -> str:
    return next((tag for tag in cell.tags if tag in TIERS), NO_TIER)


def _first_sentence(text: str) -> str:
    """The first sentence of ``text`` without a ``[self-consistency …]`` prefix, cut near :data:`_REASONING_LIMIT` characters."""
    text = _SELF_CONSISTENCY.sub('', text).strip()
    match = re.search(r'[.!?](?=\s|$)', text)
    sentence = text[: match.end()] if match else text
    return sentence if len(sentence) <= _REASONING_LIMIT else sentence[: _REASONING_LIMIT - 1].rstrip() + '…'


def _turns_text(turns: Sequence[int]) -> str:
    return f' (turn{"s" if len(turns) > 1 else ""} {", ".join(str(t) for t in turns)})' if turns else ''


def why_text(cell: Cell, *, include_content: bool, fallback: str | None = None) -> str:
    """The Why column: the judge's first finding and the turns, else the first sentence of its reasoning, with how many
    samples agreed. With content excluded only the finding kind and the turns remain."""
    detail = cell.detail
    turns = _turns_text(detail.turns)
    if not include_content:
        if not detail.kind and not detail.turns and fallback is not None:
            return fallback
        return f'{detail.kind or "violated"}{turns}'
    said = detail.why or (_first_sentence(detail.reasoning) if detail.reasoning else None)
    if said is None:
        return cell.outcome.detail if fallback is None else fallback
    agreement = f' · judges {detail.agreement[0]}/{detail.agreement[1]}' if detail.agreement else ''
    return f'{said}{turns}{agreement}'


def class_label(classes: Sequence[NotMeasuredClass]) -> str:
    """``not measured: infra, suite``, or ``not measured`` when no trial was invalid."""
    return f'not measured: {", ".join(classes)}' if classes else 'not measured'


def not_measured_reason(cell: Cell) -> str:
    """``infra: errored, provider_infra, violated in 1`` for a not-measured cell: its classes, the causes the backend named and
    the violation it still carries. Raw tokens, unescaped, for a machine reader."""
    causes = ', '.join(distinct_causes(cell))
    said = f'{", ".join(cell.outcome.classes)}: {causes}' if cell.outcome.classes and causes else ', '.join(cell.outcome.classes)
    return ', '.join(part for part in (said or _TOO_FEW_VALID, cell.outcome.detail) if part)


def _outcome_label(cell: Cell) -> str:
    outcome = cell.outcome.outcome
    return class_label(cell.outcome.classes) if outcome == 'not_measured' else outcome.replace('_', ' ')


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def _details(summary: str, body: str) -> str:
    return f'<details><summary>{summary}</summary>\n\n{body}\n\n</details>'


def _comparable(run: Run, main: Run | None) -> bool:
    """Main's run is comparable when it scored the same scenario set, eval set, env and corpus under the same schema.

    An id one of the two runs does not record cannot be compared, so a run missing any of the four is not comparable.
    """
    if main is None or main.schema != run.schema:
        return False
    kinds = ('scenarios', 'evals', 'env', 'corpus')
    return all(kind in run.suite_ids and run.suite_ids.get(kind) == main.suite_ids.get(kind) for kind in kinds)


def _new_vs_main(run: Run, main: Run) -> set[tuple[str, str]]:
    failed_on_main = {c.key for c in main.cells if c.outcome.outcome == 'failed'}
    return {c.key for c in run.cells if c.outcome.outcome == 'failed' and c.key not in failed_on_main}


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


def _heading(ctx: _Context) -> str:
    counts = ctx.counts
    parts = [f'{counts.unstable} unstable', f'{counts.advisory} advisory']
    if counts.not_measured or counts.verdict != 'not_measured':
        parts.append(f'{counts.not_measured} not measured')
    if counts.quarantine_capped:
        parts.append(f'{_plural(counts.quarantined, "scenario")} not run')
    tail = ' · '.join(parts)
    if counts.verdict == 'failed':
        shown = [_name(c) + (' (new vs main)' if ctx.new is not None and c.key in ctx.new else '') for c in ctx.failed[:_HEADING_NAMES]]
        more = f' and {len(ctx.failed) - _HEADING_NAMES} more' if len(ctx.failed) > _HEADING_NAMES else ''
        return f'## Plurai simulation · ❌ {counts.failed} failed — {", ".join(shown)}{more} · {tail}'
    if counts.verdict == 'not_measured':
        return f'## Plurai simulation · ⛔ {_not_measured_lead(counts)} · 0 failed · {tail}'
    return f'## Plurai simulation · ✅ 0 failed · {tail}'


def _not_measured_lead(counts: Tally) -> str:
    """``not measured: infra, credits (<code>, resets at …)``: every class present, with what each can say beside it."""
    credits = counts.credits
    parts = []
    for name in counts.classes:
        detail = []
        if name == 'credits' and credits is not None:
            detail = [_credits_detail(credits)]
        if name == 'suite' and counts.quarantine_capped:
            detail.append(f'{counts.quarantined} of {counts.gating_scenarios} gating scenarios quarantined as unsolvable')
        elif name == 'suite' and counts.cells == 0:
            detail.append('no scenario × policy was judged')
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
        scenarios = len({c.scenario for c in run.cells})
        first.append(f'{_md(run.selection) + ", " if run.selection else ""}{_plural(scenarios, "scenario")} × {_plural(run.trials, "trial")}')
        if run.threads is not None:
            first.append(_plural(run.threads, 'thread'))
        if run.duration_seconds is not None:
            first.append(_duration(run.duration_seconds))

    second: list[str] = []
    if compared is not None:
        second.append(f'Compared with main@{_md(compared.commit[:7])}' if compared.commit else 'Compared with main')
    if run.agent_models:
        second.append('agent model ' + ', '.join(_md(m) for m in run.agent_models))
    if run.backend_build:
        ran = f' (main ran on {_md(compared.backend_build)})' if compared is not None and compared.backend_build not in (None, run.backend_build) else ''
        second.append(f'backend build {_md(run.backend_build)}{ran}')
    stamped = [f'action {_md(run.versions["action"])}'] if run.versions.get('action') else []
    stamped += [f'SDK {_md(run.versions["sdk"])}'] if run.versions.get('sdk') else []
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
    #: The failed cells main did not fail, when main is comparable.
    new: set[tuple[str, str]] | None
    counts: Tally
    failed: tuple[Cell, ...]
    #: Every catastrophic cell violated in a trial that did not fail, whatever its outcome. A cell that could not be measured
    #: is also listed under Not measured, which holds every such cell.
    catastrophic: tuple[Cell, ...]

    @property
    def cells(self) -> tuple[Cell, ...]:
        return self.run.cells


def _context(run: Run, options: ReportOptions) -> _Context:
    main = options.main if _comparable(run, options.main) else None
    return _Context(
        run=run,
        options=options,
        include_content=resolve_include_content(options.include_content, options.visibility),
        main=main,
        new=_new_vs_main(run, main) if main is not None else None,
        counts=tally(run),
        failed=tuple(c for c in run.cells if c.outcome.outcome == 'failed'),
        catastrophic=tuple(c for c in run.cells if 'catastrophic' in c.tags and c.outcome.violated and c.outcome.outcome != 'failed'),
    )


def distinct_causes(cell: Cell) -> tuple[str, ...]:
    """The causes of the cell's invalid trials, each once, in trial order."""
    return tuple(dict.fromkeys(c for c in cell.causes if c))


def _causes_text(cell: Cell) -> str:
    return ', '.join(_md(cause) for cause in distinct_causes(cell))


def _credits_detail(credits: Credits) -> str:
    """The gateway's refusal as markdown: its code, and when the allowance resets when the gateway says. Both come from the
    backend, so both are escaped."""
    return ', '.join([_md(credits.code or 'code unknown'), *([f'resets at {_md(credits.resets_at)}'] if credits.resets_at else [])])


def _trials_text(cell: Cell) -> str:
    if cell.outcome.outcome == 'unstable':
        return cell.outcome.ratio
    causes = _causes_text(cell)
    return f'{cell.outcome.ratio} ({_outcome_label(cell)}{f" — {causes}" if causes else ""})'


def _why_cell(cell: Cell, ctx: _Context, why_limit: int | None, *, fallback: str | None = None) -> str:
    said = why_text(cell, include_content=ctx.include_content, fallback=fallback)
    if why_limit is not None and len(said) > why_limit:
        said = said[: why_limit - 1].rstrip() + '…'
    return _md(said)


def _failed_blocks(ctx: _Context, cells: Sequence[Cell] | None = None, why_limit: int | None = None) -> list[str]:
    cells = ctx.failed if cells is None else cells
    if not cells:
        return []
    new = ctx.new
    header = ['Scenario × policy', 'Tier'] + (['vs main'] if new is not None else []) + ['Why']
    rows = [
        [_name(c), _tier(c)] + (['🆕 new' if c.key in new else 'also failed on main'] if new is not None else []) + [_why_cell(c, ctx, why_limit)]
        for c in cells
    ]
    return [f'### Failed ({ctx.run.trials}/{ctx.run.trials} trials)\n{_table(header, rows)}']


def _reasoning_blocks(ctx: _Context) -> list[str]:
    """The judge's full reasoning under each failed cell; judge text, so only with content."""
    if not ctx.include_content:
        return []
    return [_details(f'{_name(c)}: judge reasoning', _fenced(c.detail.reasoning)) for c in ctx.failed if c.detail.reasoning]


def _catastrophic_block(ctx: _Context, cells: Sequence[Cell], why_limit: int | None = None) -> list[str]:
    if not cells:
        return []
    trials = ctx.run.trials
    span = f'1–{trials - 1} of {trials} trials' if trials > 1 else 'of 1 trial'
    rows = [[_name(c), _trials_text(c), _why_cell(c, ctx, why_limit, fallback='—')] for c in cells]
    title = f'unstable ({span})' if all(c.outcome.outcome == 'unstable' for c in cells) else f'violated in {span}'
    return [f'### Catastrophic, {title}\n{_table(["Scenario × policy", "Trials", "Why"], rows)}']


def _notes(ctx: _Context) -> list[str]:
    run = ctx.run
    notes = [scrub_surrogates(' '.join(n.split())) for n in ctx.options.notes]
    unspecified = sorted({c.policy for c in ctx.cells if not any(tag in TIERS for tag in c.tags)})
    notes += [f'policy {_md(policy)} has no tier: its scenario × policy are treated as advisory unless they are regressions' for policy in unspecified]
    if run.quarantined:
        notes.append(f'{_plural(len(run.quarantined), "scenario")} quarantined as unsolvable, not run: {", ".join(_md(q) for q in run.quarantined)}')
        if ctx.counts.quarantine_capped:
            notes.append(f'more than {QUARANTINE_CAP_PERCENT}% of the selected gating scenarios are quarantined, so the run is not measured: suite')
    if any(NEEDS_REDEPLOY in c.causes for c in ctx.cells):
        notes.append('the mock environment needs a redeploy and is not retried: run `/use-env`')
    notes += [f'cause {_md(cause)} is not one this report knows; its trials are counted as infra' for cause in run.unknown_causes]
    if ctx.main is None:
        reason = ctx.options.main_error or ('main ran other suite ids or another schema version' if ctx.options.main is not None else None)
        notes.append('no main run to compare' + (f' ({_md(reason)})' if reason else ''))
    return notes


def _notes_block(ctx: _Context, notes: Sequence[str] | None = None) -> list[str]:
    notes = _notes(ctx) if notes is None else notes
    return ['### Notes\n' + '\n'.join(f'- {n}' for n in notes)] if notes else []


def _attention_blocks(
    ctx: _Context, *, failed: Sequence[Cell] | None = None, catastrophic: Sequence[Cell] | None = None, why_limit: int | None = None
) -> list[str]:
    """The blocks both the report and the PR comment open with: the expiry note, the failed table and the catastrophic table.

    ``failed`` and ``catastrophic`` default to every such cell; the comment passes the rows that fit. With judge text off, one
    line under the tables says so.
    """
    tables = _failed_blocks(ctx, ctx.failed if failed is None else failed, why_limit)
    tables += _catastrophic_block(ctx, ctx.catastrophic if catastrophic is None else catastrophic, why_limit)
    blocks: list[str] = [note] if (note := _expiry_note(ctx.run, ctx.options)) else []
    return blocks + tables + ([_WITHHELD] if tables and not ctx.include_content else [])


def _detail_blocks(ctx: _Context) -> list[str]:
    """The sections after the attention blocks. A scenario × policy is listed in exactly one section: those above skip what an
    earlier one holds."""
    cells = ctx.cells
    shown = {c.key for c in (*ctx.failed, *ctx.catastrophic)}
    blocks: list[str] = []
    unstable = [c for c in cells if c.outcome.outcome == 'unstable' and c.key not in shown]
    if unstable:
        rows = [[_name(c), c.outcome.ratio, _why_cell(c, ctx, None, fallback='—')] for c in unstable]
        more = any(c.outcome.outcome == 'unstable' for c in ctx.catastrophic)
        label = f'Unstable ({len(unstable)} more)' if more else f'Unstable ({len(unstable)})'
        blocks.append(_details(label, _table(['Scenario × policy', 'Trials', 'Why'], rows)))
    advisory = [c for c in cells if c.outcome.outcome == 'advisory' and c.key not in shown]
    if advisory:
        tiers = sorted({_policy_tier(c) for c in advisory})
        suffix = '' if NO_TIER in tiers else f' tier{"s" if len(tiers) > 1 else ""}'
        rows = [[_name(c), _tier(c), c.outcome.ratio] for c in advisory]
        more = any(c.outcome.outcome == 'advisory' for c in ctx.catastrophic)
        label = f'Advisory ({len(advisory)}{" more" if more else ""}, {"/".join(tiers)}{suffix})'
        blocks.append(_details(label, _table(['Scenario × policy', 'Tier', 'Trials'], rows)))
    blocks.append(_not_measured_block(ctx))
    blocks += _notes_block(ctx)
    rows = [[_name(c), _outcome_label(c), _tier(c), c.outcome.ratio] for c in cells]
    blocks.append(_details(f'All {len(cells)} scenario × policy', _table(['Scenario × policy', 'Outcome', 'Tier', 'Violated'], rows)))
    return blocks


def _cause_words(cause: str, cell: Cell) -> str:
    """A cause in plain words with its raw token beside it for a search; a credits cause carries the gateway's code and reset."""
    if cause == 'credits' and cell.credits is not None:
        return f'{CAUSES[cause].words} ({_credits_detail(cell.credits)})'
    words = ' and '.join(CAUSES.get(part, UNKNOWN_CAUSE).words for part in cause.split('+'))
    return f'{words} ({_md(cause)})'


def _next_steps(cells: Sequence[Cell], *, capped: bool) -> str:
    """One sentence per class present: what to do about each cause in it. A joined cause takes the steps of the parts that
    earn its class."""
    steps: dict[NotMeasuredClass, list[str]] = {}
    for cell in cells:
        for cause in distinct_causes(cell):
            earned = INVALID_CLASS[classify(cause)[0]]
            for entry in (CAUSES.get(part, UNKNOWN_CAUSE) for part in cause.split('+')):
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
        causes = '; '.join(_cause_words(cause, c) for cause in distinct_causes(c))
        said = f'{", ".join(c.outcome.classes)} — {causes}' if c.outcome.classes and causes else _TOO_FEW_VALID
        detail = f', {c.outcome.detail}' if c.outcome.detail else ''
        advisory = ' (advisory, not counted)' if c.outcome.level == 'advisory' else ''
        lines.append(f'- {_name(c)}: {said}{detail}{advisory}')
    capped = ctx.counts.quarantine_capped
    if capped:
        lines += [f'- {_md(q)}: suite — {CAUSES[QUARANTINED].words}, not run' for q in ctx.run.quarantined]
    steps = _next_steps(cells, capped=capped)
    return '\n'.join(['### Not measured', *([steps] if steps else []), *(lines or ['- none'])])


def _join(heading: str, blocks: Sequence[str], footer: Sequence[str]) -> str:
    return '\n\n'.join([heading, *blocks]) + '\n\n' + '\n'.join(footer) + '\n'


def _stopped_heading(stopped: Stopped) -> str:
    return f'## Plurai simulation · ⛔ {stopped.reason}'


def _report_parts(run: Run, options: ReportOptions | None) -> tuple[str, list[str], list[str]]:
    """The report's heading, the blocks after it and its footer lines."""
    ctx = _context(run, options or ReportOptions())
    if (stopped := ctx.options.stopped) is not None:
        notes = [scrub_surrogates(' '.join(n.split())) for n in ctx.options.notes]
        return _stopped_heading(stopped), [stopped.fix, *_notes_block(ctx, notes)], _footer(run, ctx.options, compared=None)
    blocks = _attention_blocks(ctx)
    # The reasoning details sit under the failed table, before the other sections.
    blocks += _reasoning_blocks(ctx)
    blocks += _detail_blocks(ctx)
    return _heading(ctx), blocks, _footer(run, ctx.options, compared=ctx.main)


def render_report(run: Run, options: ReportOptions | None = None) -> str:
    """``report.md``: the verdict heading, the failed table, the other sections and the footer."""
    return _join(*_report_parts(run, options))


def _utf16_length(text: str) -> int:
    """``text`` in UTF-16 code units, the strictest reading of a character count."""
    return len(text.encode('utf-16-le')) // 2


def _pointer_to_results(ctx: _Context) -> str:
    """Where the full results are: the run's page when ``run_url`` is given, else the job summary, with the artifacts."""
    artifacts = ' · '.join(_md(a) for a in ctx.options.artifacts)
    if (url := ctx.options.run_url) is None:
        return 'Full results are in the job summary' + (f' and in the artifacts: {artifacts}' if artifacts else '')
    if not _WEB_ADDRESS.fullmatch(url):
        raise ValueError(f'run_url is not a plain http(s) address: {url!r}')
    return f'Full results: {url} (job summary{f"; artifacts {artifacts}" if artifacts else ""})'


def render_comment(run: Run, options: ReportOptions | None = None, *, limit_chars: int = COMMENT_LIMIT_CHARS) -> str:
    """The PR comment body: the heading, the failed table, the catastrophic block, the notes and a pointer to the full results.

    Every count in the heading has its section or is named in the pointer line. It opens with :data:`COMMENT_MARKER`, so a
    poster finds the comment again and edits it in place.

    The body stays within ``limit_chars`` (UTF-16 units). A body over it first cuts each judge reason to
    :data:`_COMMENT_WHY_LIMIT` characters, then drops rows (failed, then catastrophic, then notes) from the end, and says how
    many it dropped. A row is never cut in the middle. The heading and the pointer always stay, so a limit below their
    size cannot be met.
    """
    ctx = _context(run, options or ReportOptions())
    if (stopped := ctx.options.stopped) is not None:
        return f'{COMMENT_MARKER}\n{_stopped_heading(stopped)}\n\n{stopped.fix}\n\n{_pointer_to_results(ctx)}.\n'
    failed, catastrophic, notes = ctx.failed, ctx.catastrophic, _notes(ctx)
    total = len(failed) + len(catastrophic) + len(notes)
    counts = ctx.counts
    others = [f'{n} {name}' for n, name in ((counts.unstable, 'unstable'), (counts.advisory, 'advisory'), (counts.not_measured, 'not measured')) if n]
    where = _pointer_to_results(ctx)
    pointer = f'The job summary lists every scenario × policy counted as {", ".join(others)}.\n' if others else ''

    def compose(keep: int, why_limit: int | None) -> str:
        shown_failed = failed[:keep]
        shown_catastrophic = catastrophic[: max(0, keep - len(failed))]
        shown_notes = notes[: max(0, keep - len(failed) - len(catastrophic))]
        blocks = _attention_blocks(ctx, failed=shown_failed, catastrophic=shown_catastrophic, why_limit=why_limit)
        blocks += _notes_block(ctx, shown_notes)
        if keep < total:
            rows_dropped = len(failed) + len(catastrophic) - len(shown_failed) - len(shown_catastrophic)
            what = [f'{rows_dropped} more {"row" if rows_dropped == 1 else "rows"}'] if rows_dropped else []
            what += [_plural(len(notes) - len(shown_notes), 'note')] if len(shown_notes) < len(notes) else []
            blocks.append(f'> Truncated to fit the comment size limit: {" and ".join(what)} not shown; {_CUT_POINTER_COMMENT}.')
        return f'{COMMENT_MARKER}\n' + '\n\n'.join([_heading(ctx), *blocks]) + f'\n\n{pointer}{where}.\n'

    for why_limit in (None, _COMMENT_WHY_LIMIT):
        if _utf16_length(body := compose(total, why_limit)) <= limit_chars:
            return body
    low, high = 0, total - 1  # the most rows that fit lies in [low, high]; the body only grows with the rows kept
    while low < high:
        middle = (low + high + 1) // 2
        if _utf16_length(compose(middle, _COMMENT_WHY_LIMIT)) <= limit_chars:
            low = middle
        else:
            high = middle - 1
    return compose(low, _COMMENT_WHY_LIMIT)


def render_summary(run: Run, options: ReportOptions | None = None, *, limit_bytes: int = SUMMARY_LIMIT_BYTES) -> str:
    """The job summary body: :func:`render_report`, cut to ``limit_bytes``.

    A heading, a table and a section are never cut in the middle. ``<details>`` blocks go first, last block first, each
    replaced by a line that points to ``report.md``. If the summary is still over the limit, whole sections go from the end
    and one line points to ``report.md``. The heading and the footer always stay, so a limit below their size cannot be met.
    """
    heading, blocks, footer = _report_parts(run, options)

    def size() -> int:
        return len(_join(heading, blocks, footer).encode())

    for index in range(len(blocks) - 1, -1, -1):
        if size() <= limit_bytes:
            return _join(heading, blocks, footer)
        if blocks[index].startswith('<details>'):
            summary = _SUMMARY_TAG.search(blocks[index])
            blocks[index] = f'> {summary.group(1) if summary else "details"}: {_CUT_POINTER}.'
    if size() <= limit_bytes:
        return _join(heading, blocks, footer)
    pointer = f'> Sections after the heading are {_CUT_POINTER}.'
    while blocks and len(_join(heading, [*blocks, pointer], footer).encode()) > limit_bytes:
        blocks.pop()
    return _join(heading, [*blocks, pointer], footer)
