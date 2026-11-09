"""Client-side production-trace extraction.

Runs the COLLECTION half of the trace-analysis pipeline locally — the only part
that needs the customer's observability credentials/network. It reads a
customer's production agent traces from their observability sink (LangSmith,
Langfuse, …) and writes them as a directory of trace files that ``/analyze-traces`` uploads to
the Plurai backend; the backend holds the LLM credentials and authors + freezes
a per-corpus extraction bundle, then derives the mock-env brief.

No LLM keys are required here — collection is read-from-sink + serialize.

Mirrors ``plurai.rag_extract`` (the RAG/SQL corpus half): the customer's
credentials instantiate the sink client and are **never written** to disk — only
the resulting trace files + a manifest (source kind, project, counts; NO
credentials) are. The sink client is built with the api_key passed straight in
(no ``os.environ`` mutation), exactly like ``langsmith.build_handle``.

The wire artifacts written under ``out_dir`` are:

* ``traces/<trace_id>.json`` — one file per trace, holding that trace's WHOLE
  activity tree: for LangSmith a JSON list of run dicts (the run tree); for
  Langfuse the trace object with its full ``observations`` tree embedded (a flat
  list with ``parent_observation_id`` edges). The directory is the input the
  backend's agent-authored loader enumerates over 100%. A LangSmith export by
  ``thread_ids`` writes ``traces/<thread_id>.json`` instead: one file per
  conversation, holding every root of the thread with its run tree.
* ``manifest.json``          — source kind, project, trace count, and how the sink
  was narrowed (NO credentials).

A reader writes into ``.traces.partial/``, which becomes ``traces/`` only once the export
is complete and its manifest is written, so a failed export leaves neither a partial
``traces/`` nor a manifest.

Both registered readers are **live-verified** against a real project:

* **LangSmith**: the ``eq(trace_id, ...)`` tree query returns whole run trees (not
  root-only truncations) and every field of a live run survives serialization.
* **Langfuse**: Observations API v2 (``client.api.observations.get_many``) lists each
  trace's root and then its observation rows; the reader rebuilds the trace object
  from them, and every field of a v2 row survives serialization. A server without v2
  (self-hosted Langfuse v3) is read through the v1 trace API (``client.api.trace.list``
  + ``.get``) instead.

Offline tests hold the SERIALIZATION half, so a client bump cannot silently drop
fields. What no offline test can hold (no fake proves the server still answers):
LangSmith's tree-query filter GRAMMAR and Langfuse's v2 filter columns and read-API
routing, which depend on the live server. The backend loader is schema-free, so a bad export surfaces
only as a garbage analysis brief.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    # langsmith / langfuse ship in their own pip extras, so these are type-only
    # imports (mirrors rag_extract's TYPE_CHECKING handling of its optional deps).
    from langfuse.api import ObservationV2, TraceWithDetails, TraceWithFullDetails
    from langfuse.api.client import LangfuseAPI
    from langsmith import Client
    from langsmith.schemas import Run

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

TRACES_DIRNAME = 'traces'
MANIFEST_FILENAME = 'manifest.json'
# Where a reader writes before the export is complete; renamed to ``traces/`` only on success.
STAGING_DIRNAME = '.traces.partial'

# Source-config key NAMES that identify credentials — excluded from the manifest +
# error messages so a secret (or the name of one) never lands on disk or in a log
# line. Langfuse auth is a ``public_key`` / ``secret_key`` pair, so both names join
# the set: the VALUES never reach disk regardless (they only instantiate the client),
# but keeping their names out of ``source_config_keys`` is the same key-name hygiene.
_SECRET_KEYS = frozenset({'api_key', 'token', 'password', 'secret', 'secret_key', 'public_key'})


class TraceExtractionDependencyError(ImportError):
    """Raised when a sink's client library (e.g. ``langsmith``) isn't installed."""


class TraceExportError(ValueError):
    """An export refused for a reason its message states in full: a malformed or
    server-rejected selection, an empty result, or an out dir holding files this
    export did not write. The CLI prints it as one line, so it never carries a
    credential or a traceback's worth of client internals."""


@dataclass(frozen=True)
class _Export:
    """What a reader wrote: the file count, and how the sink was narrowed.

    ``selection`` is the reader's own account of which slice of the sink this corpus
    is; it lands in the manifest verbatim, so it never carries a credential.
    """

    trace_count: int
    selection: dict[str, Any] = field(default_factory=dict)
    # Manifest keys the reader states itself, replacing the defaults derived from the source config.
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SourceSpec:
    """How to read a sink kind + which pip extra ships its client (for the hint)."""

    reader: Callable[[dict[str, Any], Path, int | None], _Export]
    extra: str | None
    # The source-config keys that narrow WHICH traces this reader exports. A reader
    # ignores keys it does not know, so a caller checks this list (``plurai
    # capabilities``) before relying on one; otherwise it silently gets a recency export.
    selection_keys: tuple[str, ...] = ()
    # Checks the source config before the out dir is touched; a reader whose input is a directory
    # uses it to refuse an out dir that would be cleared out from under that input.
    preflight: Callable[[dict[str, Any], Path], None] | None = None


def _run_to_dict(run: Run) -> dict[str, Any]:
    """Serialize a LangSmith ``Run`` to a plain dict.

    ``Run`` is a pydantic model, but WHICH dump method it carries depends on the
    client: pydantic-v2-native schemas (langsmith 0.10.x, live-verified) expose
    ``model_dump()``, while older langsmith releases build their schemas on
    pydantic's v1 compat shim and expose only ``dict()``. Prefer ``model_dump()``
    — on a v2 model ``dict()`` still works but emits a deprecation warning per
    run (50 on a 5-trace export) and is slated for REMOVAL in pydantic v3.

    A shape carrying neither is not something we can serialize faithfully, so it
    raises: hand-listing a field subset would silently upload runs stripped of
    tokens/costs/status/events, and the schema-free backend loader cannot tell
    that apart from a genuinely sparse run.
    """
    for method in ('model_dump', 'dict'):
        serialize = getattr(run, method, None)
        if callable(serialize):
            # both return a plain dict, but the attrs are untyped across client versions.
            return cast('dict[str, Any]', serialize())
    raise TypeError(
        f'langsmith Run {getattr(run, "id", "<unknown>")!r} exposes neither model_dump() nor dict(); '
        'this client version cannot be serialized faithfully. Pin a langsmith release whose schemas are pydantic models.'
    )


def _fetch_run_tree(client: Client, project: str, trace_id: str, root: Run) -> list[Run]:
    """Fetch one trace's full run tree, degrading **visibly** to the root run.

    The ``eq(trace_id, ...)`` query is live-verified against LangSmith: it
    returns the whole tree, not the root. If it ever fails or comes back
    empty, the export still proceeds — but a root-only trace is a truncated tree
    that degrades the backend's analysis without breaking it, so it is logged at
    WARNING instead of being swallowed.
    """
    try:
        runs = list(client.list_runs(project_name=project, filter=f'eq(trace_id, "{trace_id}")'))
    except Exception:  # noqa: BLE001 — older clients / unsupported filter must not abort the whole export
        reason, trace_exc = 'fetch failed', True
    else:
        if runs:
            return runs
        reason, trace_exc = 'fetch returned 0 runs', False

    logger.warning(
        'trace %s: run-tree %s — writing the ROOT RUN ONLY. This trace is truncated and will degrade analysis.',
        trace_id,
        reason,
        exc_info=trace_exc,
    )
    return [root]


# A thread id that is already a lowercase file stem names its file as-is; any other id
# (``channel:+15550100``, one with spaces, ``@`` or an uppercase letter) gets a readable slug plus
# a hash of the id. So two ids never share a file, even on a case-insensitive file system
# (``ABC`` and ``abc``), and an id's file name never depends on the other ids listed with it.
# The manifest maps every id to its file.
_SAFE_THREAD_ID = re.compile(r'[a-z0-9][a-z0-9._-]{0,199}')

# Filter forms LangSmith is known to accept, named when it rejects one.
_FILTER_EXAMPLES = 'eq(status, "error"), has(tags, "prod"), gt(latency, 10), and(eq(feedback_key, "correctness"), eq(feedback_score, 0))'


# A trace id the sink returns is used as its file name only when it is one plain path component;
# any other id (a separator, a leading dot, a stray character) gets a slug plus a hash of the id, so
# an id from the sink can never name a path outside the export.
_SAFE_TRACE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,199}')


def _log_id(trace_id: str) -> str:
    """A server-supplied id as one inert log token: only ASCII letters, digits and ``._-`` survive, so an id cannot forge a log line."""
    return re.sub(r'[^A-Za-z0-9._-]', '?', trace_id)[:200]


def _trace_file(trace_id: str) -> str:
    if _SAFE_TRACE_ID.fullmatch(trace_id):
        return f'{trace_id}.json'
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', trace_id).strip('._-')[:60] or 'trace'
    return f'{slug}-{hashlib.sha256(trace_id.encode()).hexdigest()[:12]}.json'


def _thread_file(thread_id: str) -> str:
    if _SAFE_THREAD_ID.fullmatch(thread_id):
        return f'{thread_id}.json'
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', thread_id).strip('._-')[:60] or 'thread'
    return f'{slug}-{hashlib.sha256(thread_id.encode()).hexdigest()[:12]}.json'


def _langsmith_selection(source_config: dict[str, Any]) -> tuple[str | None, list[str] | None]:
    """Validate the optional ``filter`` / ``thread_ids`` keys, before any query is sent.

    A bare string for ``thread_ids`` would iterate as characters and an empty list
    would silently mean "the whole project", so both are refused rather than guessed.
    Repeated ids collapse to one, keeping the operator's order.
    """
    filter_ = source_config.get('filter')
    if filter_ is not None and (not isinstance(filter_, str) or not filter_.strip()):
        raise TraceExportError(f'langsmith source_config["filter"] must be a non-empty LangSmith run-filter string, e.g. {_FILTER_EXAMPLES}.')

    return filter_, _thread_ids(source_config.get('thread_ids'), 'langsmith', 'omit the key to export by recency')


def _thread_ids(raw_ids: Any, source: str, omit_hint: str) -> list[str] | None:
    """Validate a ``thread_ids`` value: a non-empty list of non-empty strings, repeats collapsed in order."""
    if raw_ids is None:
        return None
    if not isinstance(raw_ids, list):
        raise TraceExportError(f'{source} source_config["thread_ids"] must be a list of thread ids.')
    if not raw_ids:
        raise TraceExportError(f'{source} source_config["thread_ids"] must name at least one thread id ({omit_hint}).')
    bad = [t for t in raw_ids if not isinstance(t, str) or not t.strip()]
    if bad:
        raise TraceExportError(f'{source} source_config["thread_ids"] holds {len(bad)} invalid thread id(s). Each must be a non-empty string.')
    return list(dict.fromkeys(raw_ids))


@contextmanager
def _naming_a_rejected_filter(filter_: str | None) -> Iterator[None]:
    """Turn LangSmith's HTTP 400 for ``filter_`` into a named refusal.

    The client raises a plain ``LangSmithError`` for a 400 (its subclasses cover auth,
    rate limits and other statuses, which propagate unchanged). Only the filter and
    known-good forms are named: the client's message carries URLs and response bodies
    the operator does not need.
    """
    try:
        yield
    except Exception as e:
        from langsmith.utils import LangSmithError

        if filter_ and type(e) is LangSmithError and re.search(r'\b400\b', str(e)):
            raise TraceExportError(
                f'LangSmith rejected filter {filter_!r} (HTTP 400: not valid run-filter syntax). Forms it accepts include {_FILTER_EXAMPLES}.'
            ) from None
        raise


def _thread_roots(client: Client, project: str, thread_id: str, *, filter_: str | None = None, limit: int | None = None) -> list[Run]:
    """The root runs of one thread, oldest first, optionally narrowed by a run filter.

    LangSmith groups roots into a thread by the ``thread_id`` it derives from run
    metadata (``thread_id``, ``session_id`` or ``conversation_id``), and an
    ``eq(thread_id, ...)`` run filter selects on it. This is the query ``Client.read_thread``
    sends, spelled through ``list_runs`` so every client release down to the ``langsmith>=0.1.0``
    floor has it.
    """
    thread_filter = f'eq(thread_id, {json.dumps(thread_id)})'
    kwargs: dict[str, Any] = {'filter': f'and({thread_filter}, {filter_})' if filter_ else thread_filter}
    if limit is not None:
        kwargs['limit'] = limit
    with _naming_a_rejected_filter(filter_):
        return list(client.list_runs(project_name=project, is_root=True, order='asc', **kwargs))


def _trace_id(root: Run) -> str:
    return str(getattr(root, 'trace_id', None) or root.id)


def _tree_payload(client: Client, project: str, root: Run) -> list[dict[str, Any]]:
    """One root's full run tree, serialized."""
    return [_run_to_dict(r) for r in _fetch_run_tree(client, project, _trace_id(root), root)]


def _write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str))


def _extract_langsmith(source_config: dict[str, Any], traces_dir: Path, limit: int | None) -> _Export:
    """Export a LangSmith project's traces as run-tree files under ``traces_dir``.

    Two selection modes, both narrowable by ``source_config["filter"]`` (LangSmith's
    run-filter grammar):

    * **traces** (no ``thread_ids``): the newest roots matching the filter, one
      ``<trace_id>.json`` per root. ``limit`` counts roots.
    * **conversations** (``thread_ids``): one ``<thread_id>.json`` per listed thread
      holding EVERY root of that thread with its tree — the shape ``convert-trace``
      reads as one conversation. The filter decides which listed threads qualify (at
      least one root matches); a qualifying thread is still exported whole, because a
      conversation cut to its matching turns is not the conversation. ``limit`` counts
      exported conversations, in the listed order.
    """
    try:
        import langsmith
    except ImportError as e:
        # NOTE: no install command here — extract_traces appends the authoritative
        # one from the registry's `extra`, so the hint never names two ways at once.
        raise TraceExtractionDependencyError('langsmith is not installed; it is required to export LangSmith traces.') from e

    api_key = source_config.get('api_key')
    project = source_config.get('project') or source_config.get('project_name')
    if not api_key:
        raise ValueError('langsmith trace export requires source_config["api_key"] (it stays local — never written to disk).')
    if not project:
        raise ValueError('langsmith trace export requires source_config["project"] (the project/session to export traces from).')
    filter_, thread_ids = _langsmith_selection(source_config)

    # api_key passed straight to the client — NO os.environ mutation (mirrors
    # langsmith.build_handle). An optional api_url overrides the endpoint.
    client = langsmith.Client(api_key=api_key, **({'api_url': source_config['api_url']} if source_config.get('api_url') else {}))

    selection: dict[str, Any] = {
        'filter': filter_,
        'thread_ids': thread_ids,
        'limit': limit,
        'unit': 'conversation' if thread_ids else 'trace',
    }

    if thread_ids is None:
        # Each root run is one trace; pull its full tree by trace_id so the backend
        # sees the whole agent run, not just the root.
        root_query: dict[str, Any] = {'filter': filter_} if filter_ else {}
        written = 0
        with _naming_a_rejected_filter(filter_):
            for root in client.list_runs(project_name=project, is_root=True, limit=limit, **root_query):
                _write_json(traces_dir / _trace_file(_trace_id(root)), _tree_payload(client, project, root))
                written += 1
        selection['root_count'] = written
        return _Export(written, selection)

    files: dict[str, str] = {}
    unmatched: list[str] = []
    not_queried: list[str] = []
    root_count = 0
    for i, thread_id in enumerate(thread_ids):
        if limit is not None and len(files) >= limit:
            not_queried = thread_ids[i:]
            break
        # One matching root qualifies the thread; the export below is unfiltered so
        # the conversation arrives whole.
        if filter_ and not _thread_roots(client, project, thread_id, filter_=filter_, limit=1):
            unmatched.append(thread_id)
            continue
        roots = _thread_roots(client, project, thread_id)
        if not roots:
            unmatched.append(thread_id)
            continue
        files[thread_id] = _thread_file(thread_id)
        _write_json(traces_dir / files[thread_id], [run for root in roots for run in _tree_payload(client, project, root)])
        root_count += len(roots)

    if unmatched and files:
        # A listed conversation that is missing from the corpus is a narrower corpus
        # than the operator asked for; an all-missing export is refused upstream.
        # Thread ids are often phone numbers or emails, so the log carries a count; the ids are in
        # the export's local ``selection.threads_unmatched``.
        logger.warning('%d langsmith thread(s) matched no root run%s and are not exported', len(unmatched), f' for filter {filter_!r}' if filter_ else '')
    selection.update(root_count=root_count, threads_unmatched=unmatched, threads_not_queried=not_queried, thread_files=files)
    return _Export(len(files), selection)


# Page size for an unbounded (``limit=None``) Langfuse export — the trace-list
# endpoint paginates, so a full export walks pages of this many rows. The server
# also caps page size, so this is a request, not a guarantee.
_LANGFUSE_PAGE_SIZE = 50


def _langfuse_to_dict(model: Any) -> dict[str, Any]:
    """Serialize a Langfuse read-API model (a v1 trace with its observation tree, a v2 observation row, a score) to a plain dict.

    Prefer pydantic's native ``model_dump()`` — the SAME serialization convention the
    langsmith ``_run_to_dict`` path uses. For langfuse this is a MEANINGFUL choice, not
    just a stylistic one: the read-API schemas are Fern-generated and carry a custom
    ``UniversalBaseModel.dict()`` override whose output is NOT equal to ``model_dump()``
    — ``dict()`` emits the camelCase wire ALIASES (``sessionId``/``userId``/…) while
    ``model_dump()`` emits the python field names (``session_id``/``user_id``/…). The
    schema-free backend enumerates whatever key names arrive, so picking ``model_dump()``
    keeps every field name consistent with the LangSmith corpus. (Neither method is
    deprecated on these models — unlike a plain pydantic ``BaseModel.dict()`` — so the
    ``dict()`` fallback stays a safe last resort for a shape that somehow lacks
    ``model_dump()``.)

    A shape carrying neither is not something we can serialize faithfully, so it
    raises: hand-listing a field subset would silently upload traces stripped of
    observations/scores/costs, and the schema-free backend loader cannot tell that
    apart from a genuinely sparse trace.
    """
    for method in ('model_dump', 'dict'):
        serialize = getattr(model, method, None)
        if callable(serialize):
            # both return a plain dict, but the attrs are untyped across client versions.
            return cast('dict[str, Any]', serialize())
    raise TypeError(
        f'langfuse {type(model).__name__} {getattr(model, "id", "<unknown>")!r} exposes neither model_dump() nor dict(); '
        'this client version cannot be serialized faithfully. Pin a langfuse release whose schemas are pydantic models.'
    )


def _iter_recent_traces(api: LangfuseAPI, limit: int | None, *, filters: dict[str, Any] | None = None):
    """Yield trace list-rows most-recent-first, across pages, up to ``limit`` (all if None).

    ``api.trace.list`` returns one page ordered ``timestamp.desc``; this walks pages
    until the requested ``limit`` is met (or every page is exhausted when ``limit`` is
    None). Each yielded row is the LIGHTWEIGHT view (``TraceWithDetails``) — its
    ``observations`` are bare id strings; :func:`_fetch_trace_tree` re-fetches the full
    tree per row.

    ``filters`` are server-side narrowing kwargs (``environment``, ``name``) forwarded
    verbatim. Recency alone selects whatever the whole project emitted last, so a
    project that also carries the harness's own telemetry hands back traces that have
    nothing to do with the agent run being analyzed — and ``limit`` cannot separate
    them, it only decides how many of the wrong ones arrive. Narrowing on the tag the
    producer stamped is what makes the selection mean something. Absent keys are not
    passed at all, so an unfiltered call keeps the pure-recency behaviour.
    """
    remaining = limit
    page = 1
    query = {k: v for k, v in (filters or {}).items() if v}
    while True:
        page_limit = remaining if remaining is not None else _LANGFUSE_PAGE_SIZE
        resp = api.trace.list(page=page, limit=page_limit, order_by='timestamp.desc', **query)
        rows = resp.data or []
        if not rows:
            return
        yield from rows
        if remaining is not None:
            remaining -= len(rows)
            if remaining <= 0:
                return
        # Stop once the server says this was the last page (a short/empty next page
        # is also caught above); ``total_pages`` shifts with the page size but the
        # empty-rows guard makes termination robust regardless.
        total_pages = getattr(getattr(resp, 'meta', None), 'total_pages', None)
        if total_pages is not None and page >= total_pages:
            return
        page += 1


def _fetch_trace_tree(api: LangfuseAPI, trace_id: str, row: TraceWithDetails, *, truncated: list[str]) -> TraceWithFullDetails | TraceWithDetails:
    """Fetch one trace's FULL observation tree, degrading **visibly** to the list-row.

    ``api.trace.get`` returns the trace PLUS its whole observation tree (a flat list
    with ``parent_observation_id`` edges — the Langfuse analog of the LangSmith run
    tree). It is live-verified against a hosted Langfuse. If it ever fails
    (ingestion lag ⇒ ``NotFoundError``, transport, an unreadable trace) the export
    still proceeds — but the list-row's ``observations`` are only ID STRINGS, so a
    row-only trace is a truncated tree that degrades the backend's analysis without
    breaking it, and holds no conversation a converter can read. It is logged at
    WARNING, and its id is appended to ``truncated`` — which lands in the manifest's
    ``selection`` as ``truncated_trace_ids``, so the artifact itself says which files
    are rows — and one bad trace never aborts the whole export.
    """
    try:
        return api.trace.get(trace_id)
    except Exception:  # noqa: BLE001 — ingestion lag / one unreadable trace must not abort the whole export
        logger.warning(
            'trace %s: observation-tree fetch failed — writing the LIST ROW ONLY (observations as bare ids). '
            'This trace is truncated and will degrade analysis.',
            _log_id(trace_id),
            exc_info=True,
        )
        truncated.append(trace_id)
        return row


# Every field group Observations API v2 serves, so a trace file holds all Langfuse stores per observation.
_LANGFUSE_V2_FIELDS = 'core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context'
# The server's page-size ceiling on Observations API v2.
_LANGFUSE_V2_MAX_PAGE = 1000
# Observations API v2 may cut a metadata value at this many characters unless its key is
# named in ``expand_metadata``; a value this long is read again with its key expanded.
_LANGFUSE_V2_METADATA_CUT = 200
# A refused key or a missing permission fails the whole export rather than truncating every trace.
_LANGFUSE_AUTH_STATUSES = frozenset({401, 403})


def _v2_paged(fetch: Callable[..., Any], **query: Any) -> Iterator[Any]:
    """Every row of a cursor-paged Langfuse v2/v3 list endpoint."""
    cursor = None
    while True:
        page = fetch(cursor=cursor, **query)
        yield from page.data
        cursor = page.meta.cursor
        if not cursor or not page.data:
            return


def _root_filter(filters: dict[str, Any]) -> str:
    """The v2 ``filter`` for trace roots, narrowed by the export's ``environment`` / ``name`` (the trace name)."""
    conditions: list[dict[str, Any]] = [{'type': 'boolean', 'column': 'isRootObservation', 'operator': '=', 'value': True}]
    environment = filters.get('environment')
    if environment:
        values = list(environment) if isinstance(environment, (list, tuple)) else [environment]
        conditions.append({'type': 'stringOptions', 'column': 'environment', 'operator': 'any of', 'value': values})
    if filters.get('name'):
        conditions.append({'type': 'string', 'column': 'traceName', 'operator': '=', 'value': filters['name']})
    return json.dumps(conditions)


def _iter_recent_roots(api: LangfuseAPI, limit: int | None, *, filters: dict[str, Any]) -> Iterator[ObservationV2]:
    """Yield one root observation per trace, most recent first, up to ``limit`` traces (all if None).

    Langfuse v4 has no trace entity: a trace is the set of rows sharing a ``traceId``, and its
    application root (``isRootObservation``) carries the trace-level attributes. Rows always
    arrive ``startTime`` descending. A trace whose root was exported to another project is not
    listed. A trace with more than one root is listed once, by its newest.
    """
    seen: set[str] = set()
    cursor = None
    conditions = _root_filter(filters)
    while True:
        page_limit = _LANGFUSE_PAGE_SIZE if limit is None else min(limit - len(seen), _LANGFUSE_V2_MAX_PAGE)
        page = api.observations.get_many(filter=conditions, fields=_LANGFUSE_V2_FIELDS, limit=page_limit, cursor=cursor)
        for row in page.data:
            if not row.trace_id or row.trace_id in seen:
                continue
            seen.add(row.trace_id)
            yield row
            if limit is not None and len(seen) >= limit:
                return
        cursor = page.meta.cursor
        if not cursor or not page.data:
            return


def _decode_io(value: Any) -> Any:
    """An observation's ``input`` / ``output`` as v1 returned it.

    v2 returns both as the raw stored string, and Langfuse stores them JSON-encoded (a plain
    string arrives quoted), so decoding restores the value. A string that is not JSON is kept.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _observation_to_dict(row: ObservationV2) -> dict[str, Any]:
    observation = _langfuse_to_dict(row)
    for key in ('input', 'output'):
        observation[key] = _decode_io(observation.get(key))
    return observation


def _long_metadata_keys(rows: list[ObservationV2]) -> list[str]:
    """Metadata keys holding a string value long enough that v2 may have cut it."""
    keys = {k for row in rows if isinstance(row.metadata, dict) for k, v in row.metadata.items() if isinstance(v, str) and len(v) >= _LANGFUSE_V2_METADATA_CUT}
    return sorted(keys)


def _read_trace_rows(api: LangfuseAPI, trace_id: str, rows: list[ObservationV2], *, expand: list[str] | None = None) -> None:
    """Append every row of ``trace_id`` to ``rows`` (all of them, whatever their start: a reused
    trace id spans every root it was given)."""
    query: dict[str, Any] = {'trace_id': trace_id, 'fields': _LANGFUSE_V2_FIELDS, 'limit': _LANGFUSE_V2_MAX_PAGE}
    if expand:
        query['expand_metadata'] = ','.join(expand)
    rows.extend(_v2_paged(api.observations.get_many, **query))


def _fetch_trace_rows(api: LangfuseAPI, root: ObservationV2, *, truncated: list[str]) -> list[dict[str, Any]]:
    """Every observation of the root's trace, oldest first, degrading **visibly** when the read fails.

    v2 serves rows newest first; the file holds them in start order, as v1 did. If the read fails
    part way (transport, rate limit, an unreadable trace) the export keeps the rows read so far
    plus the root, logs at WARNING and lists the trace in the manifest's
    ``selection.truncated_trace_ids``: a truncated tree degrades the backend's analysis without
    breaking it, and one bad trace never aborts the whole export. A refused key does abort it.
    """
    trace_id = str(root.trace_id)
    rows: list[ObservationV2] = []
    try:
        _read_trace_rows(api, trace_id, rows)
        expand = _long_metadata_keys(rows)
        if expand:
            expanded: list[ObservationV2] = []
            _read_trace_rows(api, trace_id, expanded, expand=expand)
            rows = expanded
    except Exception as exc:  # noqa: BLE001 — one unreadable trace must not abort the whole export
        if getattr(exc, 'status_code', None) in _LANGFUSE_AUTH_STATUSES:
            raise
        logger.warning(
            'trace %s: observation read failed after %d row(s) — writing a TRUNCATED tree. This trace will degrade analysis.',
            _log_id(trace_id),
            len(rows),
            exc_info=True,
        )
        truncated.append(trace_id)
    by_id = {row.id: row for row in rows}
    by_id.setdefault(root.id, root)
    ordered = sorted(by_id.values(), key=lambda row: (row.start_time, row.id))
    return [_observation_to_dict(row) for row in ordered]


def _trace_scores(api: LangfuseAPI, trace_id: str, *, unread: list[str]) -> list[dict[str, Any]]:
    """The trace's scores (Scores API v3), or none — logged, and listed in ``selection.scores_unread_trace_ids`` — if the read fails."""
    try:
        return [_langfuse_to_dict(score) for score in _v2_paged(api.scores_v3.get_many_v3, trace_id=trace_id, fields='details,subject', limit=100)]
    except Exception:  # noqa: BLE001 — a missing score read must not abort the export of the trace itself
        logger.warning('trace %s: score read failed; the trace file holds no scores', _log_id(trace_id), exc_info=True)
        unread.append(trace_id)
        return []


def _trace_from_rows(root: ObservationV2, observations: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    """The trace object v1 returned, rebuilt from v2 rows.

    Trace-level fields come from the root observation, which carries the trace's name, session,
    user, tags, release and metadata in v4; ``input`` / ``output`` are the root's, since v4
    has no trace-level input or output. v2 reports an unset session, user, release or version
    as an empty string; the trace object has them as None, as v1 did.
    """
    trace_id = str(root.trace_id)
    head = next(o for o in observations if o['id'] == root.id)
    costs = [o['total_cost'] for o in observations if isinstance(o.get('total_cost'), (int, float))]
    project = head.get('project_id')
    return {
        'id': trace_id,
        'timestamp': head.get('start_time'),
        'name': head.get('trace_name') or head.get('name'),
        'input': head.get('input'),
        'output': head.get('output'),
        'session_id': head.get('session_id') or None,
        'user_id': head.get('user_id') or None,
        'release': head.get('release') or None,
        'version': head.get('version') or None,
        'environment': head.get('environment'),
        'tags': head.get('tags') or [],
        'public': head.get('public'),
        'bookmarked': head.get('bookmarked'),
        'metadata': head.get('metadata'),
        'latency': head.get('latency'),
        'total_cost': sum(costs, 0.0),
        'html_path': f'/project/{project}/traces/{trace_id}' if project else None,
        'scores': scores,
        'observations': observations,
    }


def _extract_langfuse(source_config: dict[str, Any], traces_dir: Path, limit: int | None) -> _Export:
    """Export a Langfuse project's traces — one ``traces/<trace_id>.json`` per trace
    (the trace plus its full observation tree).

    Reads through Observations API v2, the only trace read Langfuse Cloud serves; a server
    without it (self-hosted Langfuse v3) is read through the v1 trace API. The manifest's
    ``selection.read_api`` says which one the corpus came from."""
    try:
        import langfuse
    except ImportError as e:
        # NOTE: no install command here — extract_traces appends the authoritative
        # one from the registry's `extra`, so the hint never names two ways at once.
        raise TraceExtractionDependencyError('langfuse is not installed; it is required to export Langfuse traces.') from e

    public_key = source_config.get('public_key')
    secret_key = source_config.get('secret_key')
    if not public_key or not secret_key:
        raise ValueError(
            'langfuse trace export requires source_config["public_key"] and source_config["secret_key"] '
            '(the per-project key pair — they stay local, never written to disk).'
        )

    # Keys handed straight to the client — NO os.environ mutation (mirrors
    # observability/langfuse._build_client + the langsmith reader). An optional host
    # overrides the endpoint; when absent the SDK resolves LANGFUSE_HOST itself.
    # tracing_enabled=False: this is a READ-only export that emits no spans, so the
    # client must NOT claim the OTel global tracer provider or start a background
    # span exporter. The v3 read surface (`.api`) is unaffected by that flag.
    client = langfuse.Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        tracing_enabled=False,
        **({'host': source_config['host']} if source_config.get('host') else {}),
    )
    api = client.api

    # A Langfuse project is shared by everything that reports to it. ``environment``
    # (and ``name``) narrow the export to the traces a specific producer stamped,
    # which is the only way an export of a busy project is reproducible; without them
    # the result is "whatever this project emitted most recently". Both are optional:
    # an unfiltered export still walks the project by recency, and a filter that
    # matches nothing trips the zero-trace guard in extract_traces rather than
    # silently substituting other traces.
    filters = {'environment': source_config.get('environment'), 'name': source_config.get('name')}

    roots = _iter_recent_roots(api, limit, filters=filters)
    try:
        first = next(roots, None)
    except Exception as exc:
        if getattr(exc, 'status_code', None) != 404:
            raise
        # Observations API v2 ships with Langfuse v4; a self-hosted v3 server answers 404 and
        # still serves the v1 trace API.
        logger.warning('langfuse: the server has no Observations API v2 (a self-hosted Langfuse v3); reading through the v1 trace API')
        try:
            return _extract_langfuse_v1(api, traces_dir, limit, filters)
        except Exception as v1_exc:
            if getattr(v1_exc, 'status_code', None) != 404:
                raise
            raise TraceExportError(
                'langfuse answered 404 for both Observations API v2 and the v1 trace API; check that "host" names the Langfuse server itself.'
            ) from exc

    written = 0
    truncated: list[str] = []
    scores_unread: list[str] = []
    for root in itertools.chain([first] if first is not None else [], roots):
        trace_id = str(root.trace_id)
        observations = _fetch_trace_rows(api, root, truncated=truncated)
        trace = _trace_from_rows(root, observations, _trace_scores(api, trace_id, unread=scores_unread))
        (traces_dir / _trace_file(trace_id)).write_text(json.dumps(trace, ensure_ascii=False, default=str))
        written += 1
    # Always stated, empty lists included: "no file is a root alone" is a fact about the corpus.
    return _Export(written, {'read_api': 'v2', 'truncated_trace_ids': truncated, 'scores_unread_trace_ids': scores_unread})


def _extract_langfuse_v1(api: LangfuseAPI, traces_dir: Path, limit: int | None, filters: dict[str, Any]) -> _Export:
    """Export through the v1 trace API, for a server without Observations API v2."""
    written = 0
    truncated: list[str] = []
    # Each list-row is one trace; pull its full observation tree by id so the backend
    # sees the whole agent run, not just the trace envelope.
    for row in _iter_recent_traces(api, limit, filters=filters):
        trace_id = str(row.id)
        payload = _fetch_trace_tree(api, trace_id, row, truncated=truncated)
        (traces_dir / _trace_file(trace_id)).write_text(json.dumps(_langfuse_to_dict(payload), ensure_ascii=False, default=str))
        written += 1
    return _Export(written, {'read_api': 'v1', 'truncated_trace_ids': truncated})


_EXPORT_DIR_KEYS = frozenset({'path', 'thread_ids'})


def _export_dir_config(source_config: dict[str, Any]) -> tuple[Path, list[str] | None]:
    """Validate an ``export_dir`` source config: the directory to read and the optional ``thread_ids``.

    Keys that narrow or authenticate a live sink (``filter``, ``api_key``, …) are refused rather
    than ignored: a directory cannot be re-queried, so a filter written for it would silently select nothing.
    """
    unknown = sorted(set(source_config) - _EXPORT_DIR_KEYS)
    if unknown:
        raise TraceExportError(f'export_dir source_config takes only "path" and "thread_ids"; got {unknown}.')
    path = source_config.get('path')
    if not isinstance(path, str) or not path.strip():
        raise TraceExportError('export_dir source_config["path"] must name the directory a trace export wrote (traces/ + manifest.json).')
    root = Path(path)
    if not root.is_dir():
        raise TraceExportError(f'export_dir path {root} is not a directory.')
    return root, _thread_ids(source_config.get('thread_ids'), 'export_dir', 'omit the key to read every trace in the directory')


def _export_dir_preflight(source_config: dict[str, Any], out: Path) -> None:
    """Refuse an out dir that is, holds or sits inside the directory being read: the export clears its out dir first."""
    root, _ = _export_dir_config(source_config)
    source, target = root.resolve(), out.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise TraceExportError(f'export_dir path {root} and the out dir {out} overlap; read the export into a different out dir.')


def _inside(root: Path, entry: Path, what: str) -> Path:
    """``entry`` resolved, refused when a symlink leads out of ``root``."""
    resolved = entry.resolve()
    if not resolved.is_relative_to(root):
        raise TraceExportError(f'export_dir {what} is a symlink to {resolved}, outside {root}; only files inside the directory are read.')
    return resolved


def _export_dir_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise TraceExportError(f'export_dir path {root} holds no {MANIFEST_FILENAME}; it is not a trace export (nothing to read).')
    resolved = _inside(root, manifest_path, MANIFEST_FILENAME)
    try:
        manifest = json.loads(resolved.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} is not readable JSON ({type(e).__name__}).') from None
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA_VERSION:
        raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} is not a schema_version {SCHEMA_VERSION} trace export manifest.')
    kind = manifest.get('source_kind')
    if kind == 'export_dir' or kind not in TRACE_SOURCE_REGISTRY:
        known = ', '.join(sorted(k for k in TRACE_SOURCE_REGISTRY if k != 'export_dir'))
        raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} names source_kind {kind!r}; the sinks an export can come from are {known}.')
    if not isinstance(manifest.get('trace_count'), int) or isinstance(manifest.get('trace_count'), bool):
        raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} has no integer trace_count.')
    return manifest


def _export_dir_files(root: Path) -> dict[str, Path]:
    """File name → the file to read, for every trace file under ``traces/``; anything else there is refused."""
    traces = root / TRACES_DIRNAME
    if not traces.is_dir():
        raise TraceExportError(f'export_dir path {root} holds {MANIFEST_FILENAME} but no {TRACES_DIRNAME}/ directory.')
    _inside(root, traces, f'{TRACES_DIRNAME}/')
    files: dict[str, Path] = {}
    for entry in sorted(traces.iterdir()):
        resolved = _inside(root, entry, f'{TRACES_DIRNAME}/{entry.name}')
        if not (entry.suffix == '.json' and resolved.is_file()):
            raise TraceExportError(f'export_dir {TRACES_DIRNAME}/ in {root} holds {entry.name!r}, which is not a trace file (*.json).')
        files[entry.name] = resolved
    return files


def _root_runs(payload: Any) -> int:
    """How many root runs a trace file holds: the parentless runs of a LangSmith list, one for a Langfuse trace object."""
    if isinstance(payload, list):
        return sum(1 for run in payload if isinstance(run, dict) and run.get('parent_run_id') is None)
    return 1


def _extract_export_dir(source_config: dict[str, Any], traces_dir: Path, limit: int | None) -> _Export:
    """Read a directory ``extract_traces`` wrote and copy its trace files, byte for byte, under ``traces_dir``.

    The manifest's ``source_kind``, ``project`` and ``filters`` carry over, so everything downstream of the
    corpus (the span-tree reader, the anonymization) treats it as the live read it was taken from. Two
    selection modes, as for LangSmith:

    * **whole directory** (no ``thread_ids``): every trace file, in the manifest's thread order when the
      export is by conversation and by file name otherwise. ``limit`` counts files.
    * **conversations** (``thread_ids``): the files the export's ``thread_files`` map names for the listed
      threads, in the listed order; ``limit`` counts them. A listed thread the export holds no file for
      is recorded as unmatched. The export has to be a conversation export for this mode.

    The directory is checked first: the manifest must be a trace export manifest, its ``trace_count`` and
    ``thread_files`` must agree with the files present, and no entry may be a symlink out of the directory.
    """
    root, thread_ids = _export_dir_config(source_config)
    root = root.resolve()
    manifest = _export_dir_manifest(root)
    files = _export_dir_files(root)
    if manifest['trace_count'] != len(files):
        raise TraceExportError(
            f'export_dir {MANIFEST_FILENAME} in {root} says trace_count {manifest["trace_count"]} but {TRACES_DIRNAME}/ holds {len(files)} file(s); '
            'the directory was changed after the export or is incomplete.'
        )
    recorded = manifest.get('selection')
    original: dict[str, Any] = recorded if isinstance(recorded, dict) else {}
    thread_files = original.get('thread_files')
    if thread_files is not None:
        if not isinstance(thread_files, dict) or not all(isinstance(t, str) and isinstance(f, str) for t, f in thread_files.items()):
            raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} has a thread_files map that is not thread id → file name.')
        if sorted(thread_files.values()) != sorted(files):
            raise TraceExportError(f'export_dir {MANIFEST_FILENAME} in {root} maps threads to files that are not the files in {TRACES_DIRNAME}/.')
    if thread_ids is not None and thread_files is None:
        raise TraceExportError(
            f'export_dir {root} was exported by trace, not by conversation (its manifest has no thread_files), so "thread_ids" cannot select from it. '
            'Re-export it with thread_ids, or leave the key out.'
        )

    chosen: dict[str, str] = {}  # file name → the thread it holds ('' for a trace export)
    unmatched: list[str] = []
    not_queried: list[str] = []
    if thread_ids is not None and thread_files is not None:
        for i, thread_id in enumerate(thread_ids):
            if limit is not None and len(chosen) >= limit:
                not_queried = thread_ids[i:]
                break
            if thread_id in thread_files:
                chosen[thread_files[thread_id]] = thread_id
            else:
                unmatched.append(thread_id)
    elif thread_files is not None:
        chosen = {name: thread for thread, name in thread_files.items()}
    else:
        chosen = dict.fromkeys(files, '')
    if thread_ids is None and limit is not None:
        chosen = dict(list(chosen.items())[:limit])

    root_count = 0
    for name in chosen:
        try:
            payload = json.loads(files[name].read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            raise TraceExportError(f'export_dir {TRACES_DIRNAME}/{name} in {root} is not readable JSON ({type(e).__name__}).') from None
        root_count += _root_runs(payload)
        shutil.copyfile(files[name], traces_dir / name)

    if unmatched and chosen:
        logger.warning('export_dir holds no file for %d of the listed thread(s); they are not read', len(unmatched))
    threads = {thread: name for name, thread in chosen.items() if thread}
    selection: dict[str, Any] = {
        'filter': original.get('filter'),
        'thread_ids': thread_ids if thread_ids is not None else (list(threads) or None),
        'limit': limit,
        'unit': 'conversation' if threads else 'trace',
        'root_count': root_count,
    }
    if thread_ids is not None:
        selection.update(threads_unmatched=unmatched, threads_not_queried=not_queried)
    if threads:
        selection['thread_files'] = threads
    for key in ('truncated_trace_ids', 'scores_unread_trace_ids'):
        if isinstance(original.get(key), list):
            selection[key] = [t for t in original[key] if _trace_file(t) in chosen]
    if isinstance(original.get('read_api'), str):
        selection['read_api'] = original['read_api']
    selection['export_dir'] = {'path': str(root), 'trace_count': manifest['trace_count']}
    carried = {key: manifest[key] for key in ('source_kind', 'project', 'filters') if key in manifest}
    return _Export(len(chosen), selection, carried)


# Maps the customer-facing ``source_kind`` → its reader + the pip extra that
# ships the client. Extend with arize / braintrust as needed (the SDK already
# wraps those sinks under plurai.observability).
TRACE_SOURCE_REGISTRY: dict[str, _SourceSpec] = {
    'langsmith': _SourceSpec(_extract_langsmith, 'langsmith', ('filter', 'thread_ids')),
    'langfuse': _SourceSpec(_extract_langfuse, 'langfuse', ('environment', 'name')),
    'export_dir': _SourceSpec(_extract_export_dir, None, ('thread_ids',), _export_dir_preflight),
}


def trace_selection_keys() -> dict[str, list[str]]:
    """Per source kind, the source-config keys that narrow an export in this build."""
    return {kind: sorted(spec.selection_keys) for kind, spec in sorted(TRACE_SOURCE_REGISTRY.items())}


def _is_previous_manifest(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and {'schema_version', 'source_kind', 'trace_count'} <= data.keys()


def _clear_previous_export(out: Path, traces_dir: Path) -> None:
    """Empty ``traces_dir`` of a previous export, or refuse when it holds anything else.

    Every file under ``traces/`` is uploaded, so a re-run that left the last export in
    place would ship both, under a manifest describing only the new one. Only a
    directory this function can prove is an earlier export (flat ``*.json`` files
    beside one of our manifests) is cleared; anything else is someone's data, and the
    export stops before any query rather than deleting or mixing into it.
    """
    manifest = out / MANIFEST_FILENAME
    ours = _is_previous_manifest(manifest)
    if traces_dir.exists():
        entries = sorted(traces_dir.iterdir()) if traces_dir.is_dir() else [traces_dir]
        foreign = [e.name for e in entries if not (e.is_file() and e.suffix == '.json')]
        if entries and (foreign or not ours):
            shown = foreign or [e.name for e in entries]
            raise TraceExportError(
                f'{traces_dir} holds files that are not a previous export ({shown[:5]}{" …" if len(shown) > 5 else ""}); '
                'every file there would be uploaded with this export. Move them aside or pick another --out-dir.'
            )
        shutil.rmtree(traces_dir)
    if ours:
        manifest.unlink()


def _clear_staging(staging: Path) -> None:
    """Remove a staging folder a killed run left behind. The name is this module's own.

    ``rmtree`` refuses a symlink, so a link planted at that name raises rather than deleting
    what it points at.
    """
    if staging.exists() or staging.is_symlink():
        shutil.rmtree(staging)


def _read_export(spec: _SourceSpec, source_kind: str, source_config: dict[str, Any], traces_dir: Path, limit: int | None) -> _Export:
    """Run the reader into ``traces_dir`` and refuse an export that yielded nothing."""
    try:
        export = spec.reader(source_config, traces_dir, limit)
    except TraceExtractionDependencyError as e:
        # The reader knows WHICH client is missing; the registry knows which SDK
        # extra ships it. Join them here so the hint is actionable (parity with
        # rag_extract's per-source driver-extra hint).
        if spec.extra is None:
            raise
        raise TraceExtractionDependencyError(f'{e} It ships with the SDK extra: `pip install "plurai-simulation-sdk[{spec.extra}]"`.') from e

    # Fail fast on an empty export — a wrong project / filter / credential commonly
    # yields zero traces; without this the customer uploads an empty corpus and
    # only discovers the problem at analysis time.
    if export.trace_count == 0:
        safe_keys = sorted(k for k in source_config if k not in _SECRET_KEYS)
        narrowed = {k: export.selection[k] for k in ('filter', 'thread_ids') if export.selection.get(k)}
        if 'thread_ids' in narrowed:
            narrowed['thread_ids'] = f'{len(narrowed["thread_ids"])} thread id(s)'
        raise TraceExportError(
            f'{source_kind!r} sink yielded 0 traces from config keys {safe_keys}'
            f'{f" narrowed by {narrowed}" if narrowed else ""}. '
            f'Check the project / filter / thread ids / credentials — an empty corpus cannot be analyzed.'
        )
    return export


def extract_traces(
    source_kind: str,
    source_config: dict[str, Any],
    out_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Export production traces from one observability sink under ``out_dir``.

    Parameters
    ----------
    source_kind:
        A key of :data:`TRACE_SOURCE_REGISTRY` (e.g. ``"langsmith"``).
    source_config:
        Constructor/query kwargs for the sink (e.g. ``{"api_key": ..., "project": ...}``).
        Credentials here instantiate the client and are **never written** to disk
        — only the resulting trace files + manifest are.
    out_dir:
        Directory to write ``traces/`` + ``manifest.json`` into (created if missing).
    limit:
        Optional cap on the number of files to export (None = all): traces, or
        conversations for a LangSmith export by ``thread_ids``.

    Returns
    -------
    The manifest dict (also written to ``manifest.json``).
    """
    try:
        spec = TRACE_SOURCE_REGISTRY[source_kind]
    except KeyError as e:
        known = ', '.join(sorted(TRACE_SOURCE_REGISTRY))
        raise ValueError(f'Unknown trace source_kind {source_kind!r}. Known kinds: {known}.') from e

    if limit is not None and limit < 1:
        raise TraceExportError(f'limit must be at least 1, got {limit} (omit it to export everything the selection matches).')

    out = Path(out_dir)
    traces_dir = out / TRACES_DIRNAME
    staging = out / STAGING_DIRNAME
    if spec.preflight is not None:
        spec.preflight(source_config, out)
    _clear_previous_export(out, traces_dir)
    _clear_staging(staging)
    staging.mkdir(parents=True)

    # The reader writes into ``staging``, and only a complete, non-empty export is renamed to
    # ``traces/``. A reader that fails part-way (a rate limit on the third thread) leaves no
    # files behind: a half-written ``traces/`` without a manifest would read as someone else's
    # data and refuse every later run into this out dir.
    logger.info('Exporting traces from %s sink', source_kind)
    try:
        export = _read_export(spec, source_kind, source_config, staging, limit)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    n_traces = export.trace_count

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'source_kind': source_kind,
        # keys only — never the credential VALUES
        'source_config_keys': sorted(k for k in source_config if k not in _SECRET_KEYS),
        'project': source_config.get('project') or source_config.get('project_name'),
        # The filter VALUES, not just their key names: an auditor reading the manifest
        # has to be able to tell which slice of the project this corpus is, and the
        # key-name list alone cannot answer that. These are producer tags, never
        # credentials — the _SECRET_KEYS names stay excluded above.
        'filters': {k: source_config[k] for k in ('environment', 'name') if source_config.get(k)},
        'trace_count': n_traces,
        # How the reader narrowed the sink, and what that yielded; absent for a
        # reader that reports no selection.
        **({'selection': export.selection} if export.selection else {}),
        **export.manifest,
    }
    # The manifest lands before the rename: a run stopped between the two leaves a manifest with no
    # ``traces/``, which the next run clears, never a ``traces/`` with no manifest, which it refuses.
    manifest_path = out / MANIFEST_FILENAME
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        staging.rename(traces_dir)
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info('Exported %d trace(s) from %s → %s', n_traces, source_kind, traces_dir)
    return manifest
