"""The suite diff: what a change of committed ids changes in the suite a run scores, as markdown.

Pure and deterministic: two :class:`SuiteSide` values (the suite at the base commit and at the head) go in, a full text and a
short change list come out. The same input renders the same bytes, so the local ``plurai suite diff`` and the CI report show
one view. With ``include_content`` off the diff names scenarios, policies and tools, tiers and counts, and says "content
changed" without quoting any of it; with it on, each changed item carries a collapsible unified diff of the canonical YAML.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import yaml

from plurai.report.render import (  # noqa: PLC2701 - the one escaping and cutting rule of every rendering
    SUITE_DIFF_COMMENT_CHARS,
    SUITE_DIFF_SUMMARY_BYTES,
    _cut,
    _md,
)
from plurai.report.run import scrub_surrogates

#: Kinds in the order the diff lists them.
KINDS: Final = ('scenarios', 'evals', 'env', 'corpus')

#: Policy keys by what they mean to a reviewer. Everything else a body holds is a "setting".
_TEXT_KEYS: Final = ('prompt', 'eval_metric_description', 'description', 'expression', 'reasoning_expression')
_APPLICABILITY_KEYS: Final = ('applicability_guard', 'flow_filter', 'target_flows', 'target_complications')
_TYPE_KEYS: Final = ('type', 'evaluation_scope', 'metric_type', 'violated', 'labels')
#: Keys no verdict reads, and ones the backend rewrites on its own.
_IGNORED_KEYS: Final = frozenset({'display_name', 'prompt_summary', 'provenance', 'effective_tier'})

_DIFF_LINES: Final = 60
#: A line of a content diff is cut to this many characters, and a name in a list to this many.
_DIFF_LINE_CHARS: Final = 2_000
_NAME_CHARS: Final = 80
#: A list of names shows this many and counts the rest.
_NAMES_SHOWN: Final = 20
#: A reason or a caller's note is cut to this many characters.
_NOTE_CHARS: Final = 500
_BACKTICK_RUN: Final = re.compile('`+')
_SHORT_POINTER: Final = '_The full diff is in `suite-diff.md`._'
_DOCUMENT_LEGEND: Final = 'a name held by several corpora is compared as one document (its digests are joined)'


@dataclass(frozen=True)
class SuiteSide:
    """The suite at one commit: its ids and, when the backend could be read, its content."""

    #: ``scenarios``, ``evals``, ``env`` and ``corpus`` to the committed id (corpus ids joined with commas).
    ids: Mapping[str, str]
    #: Scenario name to ``{"inputs": ..., "metadata": ...}``.
    scenarios: Mapping[str, Mapping[str, Any]] | None = None
    #: Policy name to its body, with the backend's ``effective_tier`` added.
    policies: Mapping[str, Mapping[str, Any]] | None = None
    #: Tool name to its declaration.
    env_tools: Mapping[str, Any] | None = None
    #: Document name to the digest of its chunk texts over the committed corpora; ``{}`` for none, ``None`` when the backend
    #: lists no documents for one of them.
    corpus_documents: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SuiteDiff:
    """The rendered diff. ``moved`` is whether any committed id differs."""

    moved: bool
    #: ``suite-diff.md``: the heading, the change list and, with content, each item's collapsible diff.
    full: str
    #: ``suite-diff-comment.md``: the heading and the change list, pointing at the full diff.
    short: str


def ids_moved(base: Mapping[str, str], head: Mapping[str, str]) -> tuple[str, ...]:
    """The kinds whose committed id differs between the two commits."""
    return tuple(kind for kind in KINDS if base.get(kind) != head.get(kind))


def _short(value: str | None) -> str:
    text = (value or 'none').split(',')[0]
    return text[:8] + '…' if len(text) > 8 else text


def _canonical(value: Any) -> str:  # noqa: ANN401
    return yaml.safe_dump(value, sort_keys=True, default_flow_style=False, allow_unicode=True).rstrip('\n')


def _quoted(items: Sequence[str]) -> str:
    shown = ', '.join(_md(_cut(name, _NAME_CHARS)) for name in items[:_NAMES_SHOWN])
    return f'{shown} and {len(items) - _NAMES_SHOWN} more' if len(items) > _NAMES_SHOWN else shown


def _unified(before: Any, after: Any, label: str) -> str:  # noqa: ANN401
    lines = list(difflib.unified_diff(_canonical(before).splitlines(), _canonical(after).splitlines(), 'base', 'head', lineterm='', n=2))
    shown = [_cut(line, _DIFF_LINE_CHARS) for line in lines[:_DIFF_LINES]]
    if len(lines) > _DIFF_LINES:
        shown.append(f'… {len(lines) - _DIFF_LINES} more lines')
    body = scrub_surrogates('\n'.join(shown))
    # A fence longer than any run of backticks in the body: nothing in the body can close it and spill into the page as markup.
    fence = '`' * max([3, *(len(run) + 1 for run in _BACKTICK_RUN.findall(body))])
    return f'<details><summary>{_md(_cut(label, _NAME_CHARS * 2))}</summary>\n\n{fence}diff\n{body}\n{fence}\n\n</details>'


@dataclass
class _Item:
    """One line of the change list, and the collapsible diffs under it."""

    text: str
    details: list[str] = field(default_factory=list)


def _group(before: Mapping[str, Any], after: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if before.get(key) != after.get(key) and (key in before or key in after)]


def _policy_items(base: Mapping[str, Mapping[str, Any]], head: Mapping[str, Mapping[str, Any]], include_content: bool) -> list[_Item]:
    items: list[_Item] = []
    added, removed = sorted(set(head) - set(base)), sorted(set(base) - set(head))
    if added:
        items.append(_Item(f'policies added: {_quoted(added)}'))
    if removed:
        items.append(_Item(f'policies removed: {_quoted(removed)}'))
    for name in sorted(set(base) & set(head)):
        before, after = base[name], head[name]
        parts: list[str] = []
        details: list[str] = []
        if before.get('effective_tier') != after.get('effective_tier'):
            parts.append(f'tier {_md(str(before.get("effective_tier") or "none"))} → {_md(str(after.get("effective_tier") or "none"))}')
        for label, keys in (('text', _TEXT_KEYS), ('applicability', _APPLICABILITY_KEYS), ('type', _TYPE_KEYS)):
            changed = _group(before, after, keys)
            if changed:
                parts.append(f'{label} changed')
                if include_content:
                    details.append(_unified({k: before.get(k) for k in changed}, {k: after.get(k) for k in changed}, f'{name}: {label}'))
        known = {*_TEXT_KEYS, *_APPLICABILITY_KEYS, *_TYPE_KEYS, *_IGNORED_KEYS}
        other = sorted(k for k in (set(before) | set(after)) - known if before.get(k) != after.get(k))
        if other:
            parts.append('settings changed')
            if include_content:
                details.append(_unified({k: before.get(k) for k in other}, {k: after.get(k) for k in other}, f'{name}: settings'))
        if parts:
            items.append(_Item(f'{_md(name)}: {", ".join(parts)}', details))
    return items


def _scenario_items(base: Mapping[str, Mapping[str, Any]], head: Mapping[str, Mapping[str, Any]], include_content: bool) -> list[_Item]:
    items: list[_Item] = []
    added, removed = sorted(set(head) - set(base)), sorted(set(base) - set(head))
    if added:
        items.append(_Item(f'scenarios added: {_quoted(added)}'))
    if removed:
        items.append(_Item(f'scenarios removed: {_quoted(removed)}'))
    for name in sorted(set(base) & set(head)):
        changed: list[str] = []
        before_all: dict[str, Any] = {}
        after_all: dict[str, Any] = {}
        for group in ('inputs', 'metadata'):
            before, after = base[name].get(group) or {}, head[name].get(group) or {}
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    changed.append(key)
                    before_all[f'{group}.{key}'], after_all[f'{group}.{key}'] = before.get(key), after.get(key)
        if changed:
            names = ', '.join(_md(key) for key in dict.fromkeys(changed))
            items.append(_Item(f'{_md(name)}: {names} changed', [_unified(before_all, after_all, f'{name}: content')] if include_content else []))
    return items


def _tool_items(base: Mapping[str, Any], head: Mapping[str, Any], include_content: bool) -> list[_Item]:
    items: list[_Item] = []
    added, removed = sorted(set(head) - set(base)), sorted(set(base) - set(head))
    if added:
        items.append(_Item(f'tools added: {_quoted(added)}'))
    if removed:
        items.append(_Item(f'tools removed: {_quoted(removed)}'))
    for name in sorted(set(base) & set(head)):
        if base[name] != head[name]:
            items.append(_Item(f'{_md(name)}: tool changed', [_unified(base[name], head[name], f'{name}: tool')] if include_content else []))
    return items


def _document_items(base: Mapping[str, str], head: Mapping[str, str]) -> list[_Item]:
    """Documents by name: added, removed, and changed when the digest of the chunk texts differs, which a new text or a new
    chunking both do. No text is ever served. A legend line follows the list because a name that several corpora hold is one
    entry with their digests joined."""
    items: list[_Item] = []
    added, removed = sorted(set(head) - set(base)), sorted(set(base) - set(head))
    changed = sorted(name for name in set(base) & set(head) if base[name] != head[name])
    for label, names in (('documents added', added), ('documents removed', removed), ('document content or chunking changed', changed)):
        if names:
            items.append(_Item(f'corpus {label}: {_quoted(names)}'))
    if items:
        items.append(_Item(_DOCUMENT_LEGEND))
    return items


def _kind_items(kind: str, base: SuiteSide, head: SuiteSide, include_content: bool) -> list[_Item] | None:
    """The change list of one kind, or ``None`` when the content of either side is not known."""
    if kind == 'scenarios':
        return None if base.scenarios is None or head.scenarios is None else _scenario_items(base.scenarios, head.scenarios, include_content)
    if kind == 'evals':
        return None if base.policies is None or head.policies is None else _policy_items(base.policies, head.policies, include_content)
    if kind == 'env':
        return None if base.env_tools is None or head.env_tools is None else _tool_items(base.env_tools, head.env_tools, include_content)
    if base.corpus_documents is None or head.corpus_documents is None:
        return [_Item('corpus documents cannot be compared (the backend did not list them or they could not be read)')]
    return _document_items(base.corpus_documents, head.corpus_documents)


def render_suite_diff(
    base: SuiteSide | None,
    head: SuiteSide,
    *,
    base_ref: str | None = None,
    include_content: bool,
    base_ids: Mapping[str, str] | None = None,
    notes: Sequence[str] = (),
) -> SuiteDiff:
    """The diff of ``head`` against ``base``.

    ``base`` is ``None`` when the base suite cannot be read (the backend no longer holds it, or a fetch failed); the diff then
    lists the head's names only and says so. ``base_ids`` gives the base commit's ids when ``base`` is ``None``, so the moved
    ids are still shown. ``notes`` are lines the caller adds under the list.
    """
    base_id_map = base.ids if base is not None else (base_ids or {})
    moved = ids_moved(base_id_map, head.ids)
    where = f' (base: {_md(base_ref)})' if base_ref else ''
    lines: list[str] = []
    detail_blocks: list[str] = []
    if not moved:
        return SuiteDiff(False, '', '')
    unchanged = [kind for kind in KINDS if kind not in moved]
    for kind in moved:
        lines.append(f'- {kind} {_short(base_id_map.get(kind))} → {_short(head.ids.get(kind))}{where if kind == moved[0] else ""}')
        items = _kind_items(kind, base, head, include_content) if base is not None else None
        if base is None:
            names = _head_names(kind, head)
            lines.append('  - base suite no longer readable' + (f'; head: {names}' if names else ''))
        elif items is None:
            lines.append('  - content changed (the content could not be read)')
        elif not items:
            lines.append('  - ids differ, content equal')
        for item in items or []:
            lines.append(f'  - {item.text}')
            detail_blocks.extend(item.details)
    note_lines = [f'- {_md(_cut(note, _NOTE_CHARS))}' for note in notes]
    head_text = '### Suite changed vs base'
    # The closing lines always stay; only the changes between the heading and them are cut.
    tail = [
        *([f'- {", ".join(unchanged)}: unchanged'] if unchanged else []),
        *([] if include_content else ['- content is withheld: names, tiers and counts only']),
        *note_lines,
    ]
    # The change list is the same in both files; the short file is the list alone, the full file the list and as many collapsible
    # diffs as its share leaves room for.
    short_room = SUITE_DIFF_COMMENT_CHARS - _utf16_length(f'\n\n{_SHORT_POINTER}\n')
    short_lines = _bounded(lines, [head_text, ''], tail, short_room, _utf16_length)
    list_lines = _bounded(lines, [head_text, ''], tail, SUITE_DIFF_SUMMARY_BYTES // 2, _byte_length)
    full = _with_details(list_lines, detail_blocks, SUITE_DIFF_SUMMARY_BYTES)
    short = '\n'.join([*short_lines, '', _SHORT_POINTER]) + '\n'
    return SuiteDiff(True, full, short)


def _utf16_length(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def _byte_length(text: str) -> int:
    return len(text.encode())


def _bounded(body: Sequence[str], head: Sequence[str], tail: Sequence[str], limit: int, size: Callable[[str], int]) -> list[str]:
    """``head``, as many leading lines of ``body`` as fit within ``limit`` (as ``size`` measures the joined text with a newline
    at the end), a line counting the rest, and ``tail``. The head and the tail always stay."""

    def compose(kept: int) -> list[str]:
        omitted = len(body) - kept
        return [*head, *body[:kept], *([f'- … and {omitted} more lines not shown'] if omitted else []), *tail]

    low, high = 0, len(body)
    while low < high:
        middle = (low + high + 1) // 2
        if size('\n'.join(compose(middle)) + '\n') <= limit:
            low = middle
        else:
            high = middle - 1
    return compose(low)


def _with_details(lines: Sequence[str], blocks: Sequence[str], limit: int) -> str:
    """The change list and, after it, the collapsible diffs that fit within ``limit`` bytes, then a line counting the rest."""
    base = '\n'.join(lines) + '\n'
    used = _byte_length(base)
    kept: list[str] = []
    for block in blocks:
        cost = _byte_length(block) + 2
        # Room for the line that counts the diffs left out.
        if used + cost + 200 > limit:
            break
        kept.append(block)
        used += cost
    omitted = len(blocks) - len(kept)
    more = [f'_… {omitted} more diffs not shown (the share of the job summary for the suite diff is used up)._'] if omitted else []
    return '\n'.join([*lines, *(['', *kept, *more] if kept or more else [])]) + '\n'


def _head_names(kind: str, head: SuiteSide) -> str:
    if kind == 'scenarios' and head.scenarios is not None:
        return f'{len(head.scenarios)} scenarios'
    if kind == 'evals' and head.policies is not None:
        return f'{len(head.policies)} policies'
    if kind == 'corpus' and head.corpus_documents is not None:
        return f'{len(head.corpus_documents)} documents'
    return ''


def render_ids_only(base_ids: Mapping[str, str], head_ids: Mapping[str, str], reason: str, *, base_ref: str | None = None) -> SuiteDiff:
    """The diff when the content cannot be read at all (no token): the moved ids and the reason."""
    moved = ids_moved(base_ids, head_ids)
    if not moved:
        return SuiteDiff(False, '', '')
    where = f' (base: {_md(base_ref)})' if base_ref else ''
    lines = [f'- {kind} {_short(base_ids.get(kind))} → {_short(head_ids.get(kind))}{where if kind == moved[0] else ""}' for kind in moved]
    lines.append(f'- {_md(_cut(reason, _NOTE_CHARS))}')
    text = '\n'.join(['### Suite changed vs base', '', *lines]) + '\n'
    return SuiteDiff(True, text, text)
