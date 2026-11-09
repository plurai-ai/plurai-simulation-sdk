"""``plurai.report``: a pure core from a versioned ``run.json`` to cell outcomes and their renderings.

``read_run`` turns a ``run.json`` into cells, each judged by ``cell_outcome``; ``render_report``, ``render_summary``,
``render_comment`` and ``render_junit`` turn the run into text. ``plurai ci run`` calls them in process.
"""

from plurai.report.junit import render_junit
from plurai.report.outcome import DEFAULT_FAIL_ON_TAGS, CellOutcome, cell_outcome, level_of, tags_of
from plurai.report.render import (
    COMMENT_LIMIT_CHARS,
    COMMENT_MARKER,
    SUMMARY_LIMIT_BYTES,
    ReportOptions,
    Stopped,
    Visibility,
    render_comment,
    render_report,
    render_summary,
    resolve_include_content,
)
from plurai.report.run import KNOWN_RUN_SCHEMAS, RUN_SCHEMA, Cell, CellDetail, Run, RunJsonError, read_run
from plurai.report.tally import Tally, tally

__all__ = [
    'COMMENT_LIMIT_CHARS',
    'COMMENT_MARKER',
    'DEFAULT_FAIL_ON_TAGS',
    'KNOWN_RUN_SCHEMAS',
    'RUN_SCHEMA',
    'SUMMARY_LIMIT_BYTES',
    'Cell',
    'CellDetail',
    'CellOutcome',
    'ReportOptions',
    'Run',
    'RunJsonError',
    'Stopped',
    'Tally',
    'Visibility',
    'cell_outcome',
    'level_of',
    'read_run',
    'render_comment',
    'render_junit',
    'render_report',
    'render_summary',
    'resolve_include_content',
    'tags_of',
    'tally',
]
