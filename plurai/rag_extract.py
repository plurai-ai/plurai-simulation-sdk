"""Client-side RAG/SQL corpus extraction.

Runs the extraction half of the corpus pipeline locally: the only part that needs the customer's
data-source credentials and network (a directory, a SQLite / BigQuery / Snowflake DB, a cloud RAG
index, ...). It produces serialized *chunks* and structural *relationships* that ``/build-rag-corpus``
uploads to the Plurai simulation backend, which holds the LLM credentials and does the graph building
and testset synthesis.

No LLM keys are required here: extraction is deterministic I/O and parsing.

This depends on ``plurai-extract``, the client-shippable extraction library (source connectors and
``load_only``; nothing that needs a model). It is the ``extract`` extra of the SDK, imported lazily, so
an install without it fails with an actionable message at the first extraction. Per-source drivers are
extras of the SDK named ``extract-<driver>``::

    pip install "plurai-simulation-sdk[extract]"
    pip install "plurai-simulation-sdk[extract-bigquery]"
    pip install "plurai-simulation-sdk[extract-pdf]"

The wire artifacts written under ``out_dir`` are:

* ``chunks.json``        — ``[Document.to_dict(), …]``
* ``relationships.json`` — ``[InitialRelationship.to_dict(), …]``
* ``manifest.json``      — source kind, counts, domain hints (NO credentials)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from plurai import readers
from plurai.scan_contract.corpus import _iter_source_paths, _source_id, normalize_suffixes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from plurai_extract.sources.base import Document, InitialRelationship

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

CHUNKS_FILENAME = 'chunks.json'
RELATIONSHIPS_FILENAME = 'relationships.json'
MANIFEST_FILENAME = 'manifest.json'


class CorpusExtractionDependencyError(ImportError):
    """Raised when ``plurai-extract`` (or a per-source driver extra) isn't installed."""


_EXTRACT_INSTALL_HINT = (
    'plurai-extract is not installed. It ships as the `extract` extra of the SDK: '
    '`pip install "plurai-simulation-sdk[extract]"`. Add a driver extra for your source, '
    'e.g. `pip install "plurai-simulation-sdk[extract-pdf]"`.'
)


@dataclass(frozen=True)
class _SourceSpec:
    """How to import and which extra a source kind needs."""

    module: str
    class_name: str
    extra: str | None  # the plurai-extract extra that ships the driver; the SDK extra is `extract-<extra>`


# Maps the customer-facing ``source_kind`` → the plurai-extract source class. Keep
# these in sync with plurai-extract's ``pyproject.toml`` optional-dependency names.
SOURCE_REGISTRY: dict[str, _SourceSpec] = {
    'directory': _SourceSpec('plurai_extract.sources.directory', 'DirectorySource', 'pdf'),
    'sqlite': _SourceSpec('plurai_extract.sources.sql', 'SQLiteSource', None),
    # One record per chunk. No driver extra (stdlib json).
    'json_records': _SourceSpec('plurai_extract.sources.json_records', 'JsonRecordsSource', None),
    'bigquery': _SourceSpec('plurai_extract.sources.sql_bigquery', 'BigQuerySource', 'bigquery'),
    'snowflake': _SourceSpec('plurai_extract.sources.sql_snowflake', 'SnowflakeSource', 'snowflake'),
    'redshift': _SourceSpec('plurai_extract.sources.sql_redshift', 'RedshiftSource', 'redshift'),
    'azure_sql': _SourceSpec('plurai_extract.sources.sql_azure', 'AzureSQLSource', 'azure-sql'),
    'databricks': _SourceSpec('plurai_extract.sources.sql_databricks', 'DatabricksSource', 'databricks'),
    'azure_search': _SourceSpec('plurai_extract.sources.cloud', 'AzureAISearchSource', 'http'),
    'vertex_search': _SourceSpec('plurai_extract.sources.cloud', 'VertexAISearchSource', 'vertex'),
    'bedrock_kb': _SourceSpec('plurai_extract.sources.cloud', 'BedrockKnowledgeBaseSource', 'bedrock'),
    'langchain_vectorstore': _SourceSpec('plurai_extract.sources.cloud', 'LangChainVectorStoreSource', None),
    'llamaindex_vectorstore': _SourceSpec('plurai_extract.sources.cloud', 'LlamaIndexVectorStoreSource', None),
}

#: Read whatever the scan's scope: a scan's recorded scope may not name ``.pdf``, and a knowledge base
#: of PDFs is still that knowledge base.
_INGEST_ONLY_SUFFIXES: tuple[str, ...] = ('.pdf',)


class CorpusRootsError(ValueError):
    """The roots given cannot be ingested as one scope; the message names the roots and the fix."""


class CorpusScopeError(ValueError):
    """The scope names a suffix ingest has no reader for; the message names it."""


def ingest_scope(suffixes: Sequence[str]) -> tuple[str, ...]:
    """The suffixes a directory ingest stages: the scan's recorded scope, plus the ingest-only PDF.

    ``suffixes`` is the scope the scan admitted (``source_inventory.suffixes``, or the default set
    plus its ``--suffixes``), spelled as the scan spells it (:func:`normalize_suffixes`). A suffix in
    it that :mod:`plurai.readers` cannot read is refused by name: the scan read those files, and a
    corpus built without them would describe a different set of documents than the scan did.
    """
    scope = normalize_suffixes([*suffixes, *_INGEST_ONLY_SUFFIXES])
    if unread := [suffix for suffix in scope if not readers.readable(suffix)]:
        raise CorpusScopeError(
            f'the scan read {", ".join(unread)} file(s), and corpus ingest has no reader for {"it" if len(unread) == 1 else "them"} '
            f'(it reads {readers.describe()}). Convert those files to one of these, or re-run the scan without that suffix.'
        )
    return scope


#: Past this many suffixes, the rest of the skipped files are summed into one line.
_MAX_SKIP_WARNINGS = 10


def skipped_by_suffix_warnings(skipped: Mapping[str, int], scope: Sequence[str]) -> list[str]:
    """One warning per suffix the scope left out of the corpus, most files first.

    A corpus of one ``README.md`` beside 300 ``.html`` pages otherwise builds and reports like a corpus
    of one file. Names what was left out and the one remedy that applies: a suffix the ingest reads was
    outside the scan's scope, so widen the SCAN, then ingest again (the scan and the corpus keep
    describing the same documents); a suffix nothing reads has to be converted, since widening the scan
    to it would only get the ingest refused.
    """
    ordered = sorted(skipped.items(), key=lambda pair: (-pair[1], pair[0]))
    scope_text = ', '.join(scope)

    def warning(suffix: str, count: int) -> str:
        if readers.readable(suffix):
            return (
                f"{count} file(s) with suffix {suffix} were NOT ingested: outside the scan's scope ({scope_text}). The corpus is built "
                f'WITHOUT them — to include them, re-run the scan with --suffixes {suffix.lstrip(".")}, then ingest again'
            )
        return (
            f'{count} file(s) with suffix {suffix} were NOT ingested: corpus ingest has no reader for {suffix} (it reads {readers.describe()}). '
            'The corpus is built WITHOUT them — to include them, convert them to one of these formats'
        )

    out = [warning(suffix, count) for suffix, count in ordered[:_MAX_SKIP_WARNINGS]]
    if rest := ordered[_MAX_SKIP_WARNINGS:]:
        out.append(
            f'{sum(n for _, n in rest)} further file(s) across {len(rest)} other suffix(es) were NOT ingested ({", ".join(sfx for sfx, _ in rest[:10])})'
        )
    return out


@dataclass
class _Staged:
    """What staging the roots did: each staged file's path under the stage → its scan inventory id
    (``files``), a converted or inflated file's staged name → its own name (``renamed``), the files of
    each suffix left out of scope (``skipped_by_suffix``), the in-scope files that could not be staged
    (``not_converted``, named as the scan inventory names them, zip members as ``a.zip!/<member>``), and
    which reader the PDFs got (``pdf_reader``, set once any PDF is staged)."""

    files: dict[str, str] = field(default_factory=dict)
    renamed: dict[str, str] = field(default_factory=dict)
    skipped_by_suffix: dict[str, int] = field(default_factory=dict)
    not_converted: list[dict[str, str]] = field(default_factory=list)
    pdf_reader: str | None = None
    #: Each staged path → (the source it came from, its inventory id): a second source landing on the same
    #: path is a collision named in ``not_converted``; the same source twice (two roots cover it) is staged once.
    owners: dict[Path, tuple[tuple[Path, str], str]] = field(default_factory=dict)


def _skip_key(name: str) -> str:
    """The suffix a file left out of scope is counted under: ``.json.gz`` for a gzipped format the ingest
    reads, so the remedy names the suffix that admits it; the last suffix otherwise."""
    leaf = name.rsplit('/', 1)[-1].lower()
    if leaf.endswith(readers.GZIP_SUFFIX) and readers.kind_of(leaf) is not None:
        return Path(readers.inner_name(leaf)).suffix + readers.GZIP_SUFFIX
    return Path(leaf).suffix


def _count_skip(staged: _Staged, name: str) -> None:
    if key := _skip_key(name):
        staged.skipped_by_suffix[key] = staged.skipped_by_suffix.get(key, 0) + 1


def _admitted_paths(root: Path, scope: Sequence[str], staged: _Staged) -> Iterator[Path]:
    """The files under ``root`` the scope admits, by :func:`readers.admits` (``.json.gz`` admits
    ``x.json.gz``). The walk is the scan contract's own; it is handed ``.gz`` as well so every gzipped file
    reaches the compound-suffix test here, and the ones it fails are counted under :func:`_skip_key`."""
    for path in _iter_source_paths(root, (*scope, readers.GZIP_SUFFIX), set(), staged.skipped_by_suffix):
        if readers.admits(path.name, scope):
            yield path
        else:
            _count_skip(staged, path.name)


def _stage_member(staged: _Staged, stage: Path, path: Path, target: Path, inventory_id: str, member: str, raw: bytes, *, scope: Sequence[str]) -> None:
    """Stage one member of ``path`` (``member`` is ``''`` for the file itself) at, or beside, ``target``.

    Text is staged as itself, inflated when gzipped (``x.json.gz`` → ``x.json``); a PDF likewise when
    docling reads PDFs; every other format as its text, ``<inner name>.md``. A zip member lands under
    ``<target>!/<member>``, jailed inside the stage. Anything that cannot be staged is named in
    ``not_converted`` with the reason.
    """
    member_id = f'{inventory_id}{member}'
    if isinstance(raw, readers.Refused):
        staged.not_converted.append({'path': member_id, 'reason': raw.reason})
        return
    own = member[2:] if member else target.name
    if member:
        if not readers.admits(own, scope):
            _count_skip(staged, own)
            return
        target = target.with_name(target.name + '!') / own
        if not target.resolve().is_relative_to(stage.resolve()):
            staged.not_converted.append({'path': member_id, 'reason': 'member path escapes the stage'})
            return
    own_name = Path(own).name
    inner = readers.inner_name(own_name)
    kind = readers.kind_of(inner)
    if kind == 'pdf':
        staged.pdf_reader = readers.pdf_reader()
    as_is = kind == 'text' or (kind == 'pdf' and staged.pdf_reader == 'docling')
    final = target.with_name(inner if as_is else inner + readers.TEXT_TWIN_SUFFIX)
    source = (path.resolve(), member)
    if owner := staged.owners.get(final):
        if owner[0] != source:
            staged.not_converted.append({'path': member_id, 'reason': f'name collides with {owner[1]}'})
        return
    if os.path.lexists(final):  # a staged path no source claimed: never overwritten
        staged.not_converted.append({'path': member_id, 'reason': f'name collides with {final.relative_to(stage).as_posix()}'})
        return
    if as_is and not member and inner == own_name:
        content: bytes | str | None = None  # the file itself, linked
    elif as_is:
        try:
            content = readers.decompress(own_name, raw)[1]
        except readers.UnreadableError as exc:
            staged.not_converted.append({'path': member_id, 'reason': str(exc)})
            return
    else:
        try:
            read = readers.to_text(own_name, raw)
        except readers.ConverterMissingError as exc:
            raise CorpusExtractionDependencyError(str(exc)) from exc
        if read.text is None:
            staged.not_converted.append({'path': member_id, 'reason': read.reason})
            return
        content = read.text
    final.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        try:
            final.symlink_to(path)
        except OSError:  # a filesystem without symlinks: a copy reads the same
            shutil.copy2(path, final)
    elif isinstance(content, str):
        final.write_text(content, encoding='utf-8')
    else:
        final.write_bytes(content)
    if final.name != own_name:
        staged.renamed[final.name] = own_name
    staged.owners[final] = (source, member_id)
    staged.files[final.relative_to(stage).as_posix()] = member_id


def _stage_roots(roots: Sequence[str | Path], stage: Path, *, scope: Sequence[str]) -> _Staged:
    """Stage every file of ``scope`` the roots cover into ``stage``, at its path under the roots' common parent.

    One extractor load over ``stage`` then gives ONE corpus for all the roots. Keeping the relative
    layout keeps markdown links between covered files resolvable, and a file two roots both cover is
    staged once. Each file is read through :func:`readers.members` and staged by :func:`_stage_member`:
    a zip's members one by one, a gzip inflated, a converted format as its markdown. Records each staged
    file's name in the scan contract's ``source_inventory`` — the path under its root, prefixed with the
    root's directory name when there are several roots (``scan_contract.corpus.build_source_inventory``'s
    rule), ``!/<member>`` appended for a zip member — and, per suffix, how many files the scope left out.
    """
    given = [Path(r).expanduser() for r in roots]
    resolved = [r.resolve() for r in given]
    if missing := [str(r) for r in resolved if not r.exists()]:
        raise FileNotFoundError(f'corpus root(s) not found: {missing}')
    # A root is named as the scan names it: by the path it was given, not a symlink's target. Two roots
    # with one name would give their files one inventory id, which the scan refuses; so does the ingest.
    names = [r.name for r in given]
    if clash := next((i for i, n in enumerate(names) if n in names[:i]), None):
        first = names.index(names[clash])
        a, b = str(roots[first]), str(roots[clash])  # as the operator typed them
        if resolved[first] == resolved[clash]:
            raise CorpusRootsError(f'corpus root {a} is given twice ({a} and {b}). Pass it once.')
        raise CorpusRootsError(
            f'corpus roots {a} and {b} are both named {names[clash]!r}, so their files would share '
            'inventory ids and one would shadow the other. Pass their common parent as one root.'
        )
    base = Path(os.path.commonpath([r if r.is_dir() else r.parent for r in resolved]))
    prefixed = len(resolved) > 1
    staged = _Staged()
    for name, root in zip(names, resolved, strict=True):
        for path in _admitted_paths(root, scope, staged):
            target = stage / path.relative_to(base)
            if root.is_file():
                target = target.with_name(name)
            source_id = name if root.is_file() else _source_id(root, path)
            inventory_id = f'{name}/{source_id}' if prefixed and not root.is_file() else source_id
            for member, raw in readers.members(path):
                _stage_member(staged, stage, path, target, inventory_id, member, raw, scope=scope)
    return staged


def _import_load_only():  # noqa: ANN202 - dynamic import guard
    try:
        from plurai_extract.loader import load_only
    except ImportError as e:
        raise CorpusExtractionDependencyError(_EXTRACT_INSTALL_HINT) from e
    return load_only


def _import_source_class(source_kind: str):  # noqa: ANN202 - dynamic import guard
    try:
        spec = SOURCE_REGISTRY[source_kind]
    except KeyError as e:
        known = ', '.join(sorted(SOURCE_REGISTRY))
        raise ValueError(f'Unknown source_kind {source_kind!r}. Known kinds: {known}.') from e

    import importlib

    try:
        module = importlib.import_module(spec.module)
    except ImportError as e:
        # Reaching here means the plurai-extract PACKAGE (or the source module
        # itself) failed to import — NOT a missing per-source driver (those import
        # lazily and are handled in extract_corpus). Distinguish the common case
        # (package absent) from any other import failure, and surface the cause.
        if (e.name or '').split('.')[0] == 'plurai_extract':
            detail = _EXTRACT_INSTALL_HINT
        else:
            detail = f'failed to import the {source_kind!r} source ({e}).'
        raise CorpusExtractionDependencyError(detail) from e
    return getattr(module, spec.class_name)


def extract_corpus(
    source_kind: str,
    source_config: dict[str, Any],
    out_dir: str | Path,
    *,
    domain: dict[str, Any] | None = None,
    load_kwargs: dict[str, Any] | None = None,
    roots: Sequence[str | Path] = (),
    suffixes: Sequence[str] | None,
) -> dict[str, Any]:
    """Extract a corpus from one data source and persist it under ``out_dir``.

    Parameters
    ----------
    source_kind:
        A key of :data:`SOURCE_REGISTRY` (e.g. ``"directory"``, ``"sqlite"``,
        ``"bigquery"``).
    source_config:
        Constructor kwargs for the plurai-extract source (e.g. ``{"path": "./docs"}``,
        ``{"db_dir": "/data"}``, or cloud endpoint + api key). Credentials in
        here are used to instantiate the source and are **never written** to
        disk — only the resulting chunks/relationships are.
    out_dir:
        Directory to write ``chunks.json`` / ``relationships.json`` /
        ``manifest.json`` into (created if missing).
    domain:
        Optional domain hints the backend turns into a ``DomainConfig``
        (e.g. ``{"task_description": "...", "semantic_enrichment": true}``).
        Persisted verbatim in the manifest.
    load_kwargs:
        Optional kwargs forwarded to ``source.load()`` (e.g. chunk sizing).
    roots:
        ``directory`` only: every root of the knowledge scope, each a directory or one file, all
        extracted into ONE corpus. ``source_config["path"]`` is the one-root spelling of the same
        thing; pass one or the other.
    suffixes:
        ``directory``: REQUIRED — the suffix scope the scan read these roots with (its recorded
        ``source_inventory.suffixes``). The ingest stages exactly that scope (:func:`ingest_scope`),
        refuses a suffix in it that it has no reader for, and names, per suffix, the files the scope
        left out. ``None`` for every other source kind.

    Returns
    -------
    The manifest dict (also written to ``manifest.json``).
    """
    load_only = _import_load_only()
    source_cls = _import_source_class(source_kind)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = dict(source_config)
    staged: _Staged | None = None
    scope: tuple[str, ...] = ()
    if source_kind == 'directory':
        if bool(roots) == ('path' in config):
            raise ValueError('a directory source takes its roots OR source_config["path"]: exactly one of them')
        if suffixes is None:
            raise ValueError('a directory source needs the suffix scope the scan read its roots with (suffixes=)')
        scope = ingest_scope(suffixes)
        roots = list(roots) or [config.pop('path')]
        # The extractor reads the PDFs, plus as text every other staged file: a converted one is staged as markdown,
        # a gzipped one under its inner name.
        config['text_suffixes'] = tuple(sorted({'.md', *(Path(readers.inner_name(f'x{s}')).suffix for s in scope if readers.kind_of(f'x{s}') == 'text')}))
    elif roots:
        raise ValueError(f'roots apply to a directory source, not to {source_kind!r}')
    elif suffixes is not None:
        raise ValueError(f'suffixes apply to a directory source, not to {source_kind!r}')

    logger.info('Extracting corpus from %s source', source_kind)
    # The per-source DRIVER (e.g. google-cloud-bigquery) is imported lazily when
    # the source is constructed / loaded, NOT when its module is imported — so a
    # missing driver surfaces HERE, not in _import_source_class. Re-wrap it into
    # the same actionable dependency error (with the install hint) so the user
    # gets "add plurai-simulation-sdk[extract-bigquery]" instead of an opaque ModuleNotFound. A plain ImportError counts too:
    # plurai-extract raises one when its PDF reader (docling) cannot be imported.
    try:
        with tempfile.TemporaryDirectory(prefix='plurai-corpus-roots-') as stage:
            if roots:
                staged = _stage_roots(roots, Path(stage), scope=scope)
                config['path'] = stage
            source = source_cls(**config)
            documents, relationships = load_only(source, **(load_kwargs or {}))
    except CorpusExtractionDependencyError:
        raise
    except ImportError as e:
        spec = SOURCE_REGISTRY.get(source_kind)
        hint = f' Install the driver with `pip install "plurai-simulation-sdk[extract-{spec.extra}]"`.' if spec and spec.extra else ''
        raise CorpusExtractionDependencyError(f'The {source_kind!r} source needs a driver that is not installed ({e.name or e}).{hint}') from e

    # Fail fast on an empty extraction. A wrong path / glob / query / credential
    # commonly yields zero documents; without this the customer would upload an
    # empty corpus and only discover the problem at build time.
    if not documents:
        left_out = ' '.join(skipped_by_suffix_warnings(staged.skipped_by_suffix, scope)) if staged is not None else ''
        raise ValueError(
            f'{source_kind!r} source loaded 0 documents from config keys '
            f'{sorted(source_config)}. Check the path / query / credentials — '
            f'an empty corpus cannot be built.' + (f' {left_out}.' if left_out else '')
        )

    if staged:
        files, renamed = staged.files, staged.renamed

        # The extractor names a file by its path under the stage only when two files share its name; that
        # path runs through every directory above the roots, so it is uploaded as the scan inventory id.
        # A converted file keeps its own name, not its staged markdown's.
        def uploaded(name: str) -> str:
            return files.get(name, name) if '/' in name else renamed.get(name, name)

        for doc in documents:
            doc.name = uploaded(doc.name)
        for rel in relationships:
            rel.source_doc_name, rel.target_doc_name = uploaded(rel.source_doc_name), uploaded(rel.target_doc_name)
            if 'target_file' in rel.properties:
                rel.properties['target_file'] = uploaded(rel.properties['target_file'])

    chunks: list[Document] = documents
    rels: list[InitialRelationship] = relationships

    (out / CHUNKS_FILENAME).write_text(json.dumps([d.to_dict() for d in chunks], ensure_ascii=False))
    (out / RELATIONSHIPS_FILENAME).write_text(json.dumps([r.to_dict() for r in rels], ensure_ascii=False))

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'source_kind': source_kind,
        # keys only — never the credential VALUES
        'source_config_keys': sorted(source_config),
        'document_count': len(chunks),
        'relationship_count': len(rels),
        'domain': domain or {},
        # A directory ingest: the suffixes it stages, how many files it staged, and each one the extractor
        # turned into no chunk (or that could not be converted to text), with the reason, named as the scan
        # inventory names it. The build's reconcile excuses only these. A file read from only some of its
        # pages is named apart, with the pages it gave no chunk from (`pages` of `of`). Per suffix, the
        # files the scope left out, and the warning that names them. Which reader the PDFs got, when any was
        # staged: markitdown's has no per-page report, so a PDF it read only partly is not named partial.
        **({'pdf_reader': staged.pdf_reader} if staged is not None and staged.pdf_reader else {}),
        **(
            {
                'staged_suffixes': list(scope),
                'staged_files': len(staged.files) + len(staged.not_converted),
                'not_ingested': [*staged.not_converted, *({**skip, 'path': staged.files[skip['path']]} for skip in source.skipped_files)],
                'partially_ingested': [{**part, 'path': staged.files[part['path']]} for part in source.partial_files],
                'skipped_by_suffix': dict(sorted(staged.skipped_by_suffix.items())),
                'skipped_suffix_warnings': skipped_by_suffix_warnings(staged.skipped_by_suffix, scope),
            }
            if staged is not None
            else {}
        ),
        'files': {
            'chunks': CHUNKS_FILENAME,
            'relationships': RELATIONSHIPS_FILENAME,
        },
    }
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    logger.info(
        'Extracted %d chunks + %d relationships → %s',
        len(chunks),
        len(rels),
        out,
    )
    return manifest
