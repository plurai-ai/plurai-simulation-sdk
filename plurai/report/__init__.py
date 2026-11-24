"""``plurai.report``: a pure core from a versioned ``run.json`` to cell outcomes and their renderings.

``read_run`` turns a ``run.json`` into cells, each judged by ``cell_outcome``; ``render_report``, ``render_summary``,
``render_comment`` and ``render_junit`` turn the run into text. ``plurai ci run`` calls them in process.
"""

from plurai.report.content import run_json_for_upload, without_content
from plurai.report.junit import render_junit
from plurai.report.outcome import DEFAULT_FAIL_ON_TAGS, CellOutcome, cell_outcome, level_of, tags_of
from plurai.report.render import (
    CHECK_RUN_LIMIT_CHARS,
    COMMENT_LIMIT_CHARS,
    COMMENT_MARKER,
    SUMMARY_LIMIT_BYTES,
    ReportOptions,
    Stopped,
    Visibility,
    ignored_regressions,
    is_plain_web_address,
    override_note_text,
    render_comment,
    render_heading,
    render_report,
    render_summary,
    resolve_include_content,
)
from plurai.report.run import (
    KNOWN_RUN_SCHEMAS,
    RUN_SCHEMA,
    Cell,
    CellDetail,
    Ignored,
    NotRun,
    OverrideNote,
    Run,
    RunJsonError,
    read_run,
)
from plurai.report.suite_diff import SuiteDiff, SuiteSide, ids_moved, render_ids_only, render_suite_diff
from plurai.report.tally import QUARANTINE_CAP_PERCENT, Tally, newly_quarantined, over_quarantine_cap, tally

__all__ = [
    'CHECK_RUN_LIMIT_CHARS',
    'COMMENT_LIMIT_CHARS',
    'COMMENT_MARKER',
    'DEFAULT_FAIL_ON_TAGS',
    'KNOWN_RUN_SCHEMAS',
    'QUARANTINE_CAP_PERCENT',
    'RUN_SCHEMA',
    'SUMMARY_LIMIT_BYTES',
    'Cell',
    'CellDetail',
    'CellOutcome',
    'Ignored',
    'NotRun',
    'OverrideNote',
    'ReportOptions',
    'Run',
    'RunJsonError',
    'Stopped',
    'SuiteDiff',
    'SuiteSide',
    'Tally',
    'Visibility',
    'cell_outcome',
    'ignored_regressions',
    'ids_moved',
    'is_plain_web_address',
    'level_of',
    'newly_quarantined',
    'override_note_text',
    'over_quarantine_cap',
    'read_run',
    'render_comment',
    'render_heading',
    'render_ids_only',
    'render_junit',
    'render_report',
    'render_suite_diff',
    'render_summary',
    'run_json_for_upload',
    'resolve_include_content',
    'tags_of',
    'tally',
    'without_content',
]
