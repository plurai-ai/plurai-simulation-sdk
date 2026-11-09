"""The one home of file-format knowledge: which suffixes the SDK reads, how, and the archive layer under them.

Every SDK path that reads a customer's files asks this module, so a format is added or changed in one place:

* :data:`FORMATS` — the suffix table. ``text`` is read as UTF-8; ``records`` (``.jsonl``/``.ndjson``) is
  rendered one ``# record <n>`` section per line; ``convert`` goes through markitdown's converter for that
  format; ``pdf`` goes through docling where plurai-extract has it, else markitdown's PDF reader.
* The compression layer — ``<suffix>.gz`` is read as ``<suffix>`` once inflated. Compression never changes
  the reading: ``x.json.gz`` reads exactly as ``x.json`` does.
* The archive layer — a ``.zip`` is read member by member, each by its own suffix, under :data:`LIMITS`.
  Member ids are ``!/<member>``, appended to the archive's own id (``a.zip!/dir/m.txt``). One level only: a
  ``.zip`` or ``.gz`` inside a zip is named, never opened.

Bytes are hashed as stored (a gz file's compressed bytes, a zip member's own bytes); text is converted from
them. A member that is not read is still yielded, as :class:`Refused`, so every caller names it.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import os
import posixpath
import re
import stat
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from plurai.scan_contract.corpus import decode_source

Kind = Literal['text', 'records', 'convert', 'pdf']

#: How each readable suffix is read. A suffix outside this table has no reader. The text formats come
#: FIRST and in the vendored checker's own order (``scan_contract.corpus.DEFAULT_SOURCE_SUFFIXES``), so
#: :data:`SCAN_SUFFIXES` is upstream's default WIDENED — the same rule ``--suffixes`` follows — and
#: the checker's ``--suffixes <the rest>`` option records the identical scope.
FORMATS: Mapping[str, Kind] = {
    **dict.fromkeys(('.md', '.markdown', '.txt', '.yaml', '.yml', '.json', '.rst'), 'text'),
    **dict.fromkeys(('.jsonl', '.ndjson'), 'records'),
    **dict.fromkeys(('.html', '.htm', '.csv', '.docx', '.pptx', '.xlsx'), 'convert'),
    '.pdf': 'pdf',
}

GZIP_SUFFIX = '.gz'
ZIP_SUFFIX = '.zip'

#: Every suffix a scan or a corpus reads by default: each format, each format gzipped, and zip archives.
SCAN_SUFFIXES: tuple[str, ...] = (*FORMATS, *(f'{suffix}{GZIP_SUFFIX}' for suffix in FORMATS), ZIP_SUFFIX)

#: The suffix of a text twin: the converted text written beside (never into) a binary's own folder.
TEXT_TWIN_SUFFIX = '.md'


@dataclass(frozen=True)
class Limits:
    """How much one archive may inflate. Reads are streamed and counted, so a cap holds whatever the ratio."""

    member_bytes: int
    archive_bytes: int
    members: int


LIMITS = Limits(member_bytes=64 * 1024 * 1024, archive_bytes=256 * 1024 * 1024, members=10_000)

#: Raises :attr:`Limits.archive_bytes` for an operator whose archives are larger; the other caps are fixed.
ARCHIVE_BYTES_ENV = 'PLURAI_ARCHIVE_MAX_BYTES'


def limits() -> Limits:
    """:data:`LIMITS`, with ``archive_bytes`` from :data:`ARCHIVE_BYTES_ENV` when it is set."""
    raw = os.environ.get(ARCHIVE_BYTES_ENV, '').strip()
    if not raw:
        return LIMITS
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f'{ARCHIVE_BYTES_ENV} must be a positive whole number of bytes, got {raw!r}.')
    return Limits(member_bytes=LIMITS.member_bytes, archive_bytes=int(raw), members=LIMITS.members)


class Refused(bytes):
    """A file or member that is named but not read, with the reason. Its bytes are what was stored (empty
    when nothing could be read safely), so a caller that hashes still can."""

    reason: str

    def __new__(cls, raw: bytes = b'', *, reason: str) -> Refused:
        self = super().__new__(cls, raw)
        self.reason = reason
        return self


@dataclass(frozen=True)
class TextRead:
    """One file's text, or ``None`` and the reason it has none."""

    text: str | None
    reason: str = ''


class ConverterMissingError(ImportError):
    """markitdown, the ``readers`` extra of the SDK, is not installed: no file of a converted format can be read."""


class UnreadableError(ValueError):
    """The compression layer could not be read; the message is the reason."""


def _leaf(name: str) -> str:
    """The last path segment of a file or member id, lower-cased: ``a.zip!/d/M.TXT`` -> ``m.txt``."""
    return posixpath.basename(name.rsplit('!/', 1)[-1].replace('\\', '/')).lower()


def inner_name(name: str) -> str:
    """``name`` without its compression suffix: ``x.json.gz`` -> ``x.json``; anything else unchanged."""
    return name[: -len(GZIP_SUFFIX)] if name.lower().endswith(GZIP_SUFFIX) else name


def _format_suffix(name: str) -> str:
    """The suffix that picks the reader: the leaf's last suffix, ``''`` when it has none."""
    return Path(_leaf(name)).suffix


def kind_of(name: str) -> Kind | None:
    """How ``name`` is read once inflated, ``None`` when it has no reader (an archive itself included)."""
    return FORMATS.get(_format_suffix(inner_name(name)))


def admits(name: str, scope: Sequence[str]) -> bool:
    """Whether ``name`` falls in ``scope``: its lower-cased leaf ends with one of the scope's suffixes, so
    ``.json.gz`` admits ``x.json.gz`` where a ``Path.suffix`` test sees only ``.gz``. A name that IS the
    suffix (a dotfile ``.json``) is not admitted, as the scan contract's walk rules."""
    leaf = _leaf(name)
    return any(leaf.endswith(s) and leaf != s for s in (suffix.lower() for suffix in scope))


def readable(suffix: str) -> bool:
    """Whether a scope suffix (``.docx``, ``.json.gz``, ``.zip``) names files this module can read."""
    suffix = suffix.lower()
    return suffix == ZIP_SUFFIX or kind_of(f'x{suffix}') is not None


def describe() -> str:
    """The formats this module reads, as one phrase for an operator message."""
    return f'{", ".join(FORMATS)}, each also gzipped ({GZIP_SUFFIX}), and {ZIP_SUFFIX} archives'


def _read_capped(stream: io.BufferedIOBase | gzip.GzipFile, cap: int) -> bytes | None:
    """Up to ``cap`` bytes of ``stream``, or ``None`` once it holds more: one byte past the cap is the most
    ever inflated."""
    data = stream.read(cap + 1)
    return None if len(data) > cap else data


def decompress(name: str, raw: bytes, *, cap: int | None = None) -> tuple[str, bytes]:
    """``(inner name, bytes)``: a ``.gz`` inflated (at most ``cap`` bytes, default the member cap), anything
    else as given. Raises :class:`UnreadableError` naming why a gzip cannot be read."""
    if not name.lower().endswith(GZIP_SUFFIX):
        return name, raw
    inner = inner_name(name)
    if _leaf(inner).endswith((GZIP_SUFFIX, ZIP_SUFFIX)):
        raise UnreadableError('nested archive, not opened')
    cap = limits().member_bytes if cap is None else cap
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as stream:
            data = _read_capped(stream, cap)
    except (EOFError, OSError, zlib.error) as exc:
        raise UnreadableError(f'truncated or corrupt gzip ({type(exc).__name__})') from exc
    if data is None:
        raise UnreadableError(f'inflates past {cap} bytes, not read')
    return inner, data


# --- zip ------------------------------------------------------------------------------------------------

#: Entries that are archive furniture, not content: skipped without a word.
_JUNK_NAMES = frozenset({'.ds_store', 'thumbs.db'})
_DRIVE = re.compile(r'^[A-Za-z]:')


def _unsafe_name(name: str) -> str | None:
    """Why a member name cannot be a path under the archive's id, or ``None`` when it can."""
    if '\\' in name:
        return 'backslash in member name'
    if name.startswith('/'):
        return 'absolute member path'
    if _DRIVE.match(name):
        return 'drive letter in member path'
    if '..' in name.split('/'):
        return 'member path escapes the archive (..)'
    return None


def _is_junk(name: str) -> bool:
    return name.endswith('/') or name.startswith('__MACOSX/') or posixpath.basename(name).lower() in _JUNK_NAMES


def _zip_members(path: Path, caps: Limits) -> Iterator[tuple[str, bytes]]:
    stored = path.read_bytes()
    try:
        archive = zipfile.ZipFile(io.BytesIO(stored))
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        yield '', Refused(stored, reason=f'corrupt archive ({type(exc).__name__})')
        return
    with archive:
        infos = archive.infolist()
        if len(infos) > caps.members:
            yield '', Refused(stored, reason=f'{len(infos)} members, more than the {caps.members} an archive may hold; not opened')
            return
        seen: set[str] = set()
        total = 0
        for info in infos:
            name = info.filename
            if _is_junk(name):
                continue
            member = f'!/{name}'
            if why := _unsafe_name(name):
                yield member, Refused(reason=why)
                continue
            normalized = posixpath.normpath(name)
            if normalized in seen:
                yield member, Refused(reason='duplicate member name')
                continue
            seen.add(normalized)
            if stat.S_ISLNK(info.external_attr >> 16):
                yield member, Refused(reason='symlink member, not followed')
                continue
            if info.flag_bits & 0x1:
                yield member, Refused(reason='encrypted member, not read')
                continue
            if _leaf(name).endswith((GZIP_SUFFIX, ZIP_SUFFIX)):
                yield member, Refused(reason='nested archive, not opened')
                continue
            if total >= caps.archive_bytes:
                yield member, Refused(reason=f'the archive passed its {caps.archive_bytes}-byte cap before this member; not read')
                continue
            cap = min(caps.member_bytes, caps.archive_bytes - total)
            try:
                with archive.open(info) as stream:
                    data = _read_capped(stream, cap)
            except (zipfile.BadZipFile, OSError, EOFError, zlib.error, NotImplementedError, RuntimeError) as exc:
                yield member, Refused(reason=f'unreadable member ({type(exc).__name__})')
                continue
            if data is None:
                total = caps.archive_bytes if cap < caps.member_bytes else total
                limit = 'the member cap' if cap == caps.member_bytes else "the archive's remaining cap"
                yield member, Refused(reason=f'inflates past {cap} bytes ({limit}), not read')
                continue
            total += len(data)
            yield member, data


def members(path: str | os.PathLike[str], *, caps: Limits | None = None) -> Iterator[tuple[str, bytes]]:
    """``(member id, bytes)`` for each part of ``path`` that is read on its own.

    A plain file, a ``.gz`` included, is one member ``('', its stored bytes)``: compressed bytes are hashed as
    stored and inflated by :func:`to_text`. A ``.zip`` gives ``('!/<name>', bytes)`` per member, read from
    memory, never extracted by name. A member that is not read — an unsafe or duplicate name, a symlink, an
    encrypted or nested-archive member, one past a cap — is yielded as :class:`Refused` with its reason; an
    archive that cannot be opened at all, or holds more than ``caps.members`` entries, is one
    ``('', Refused(stored bytes))``. Directory entries, ``__MACOSX/``, ``.DS_Store`` and ``Thumbs.db`` are skipped.
    """
    path = Path(path)
    if path.name.lower().endswith(ZIP_SUFFIX):
        yield from _zip_members(path, caps or limits())
        return
    yield '', path.read_bytes()


# --- converters -----------------------------------------------------------------------------------------

#: The markitdown converter for each converted suffix. Per format, never ``MarkItDown()``: its fallback chain
#: guesses (magika) and re-reads a corrupt ``.docx`` as zip XML instead of refusing it.
_CONVERTERS: Mapping[str, str] = {
    '.html': 'HtmlConverter',
    '.htm': 'HtmlConverter',
    '.csv': 'CsvConverter',
    '.docx': 'DocxConverter',
    '.pptx': 'PptxConverter',
    '.xlsx': 'XlsxConverter',
    '.pdf': 'PdfConverter',
}


def _markitdown_convert(name: str, data: bytes) -> str:
    try:
        from markitdown import StreamInfo, converters
    except ImportError as exc:
        raise ConverterMissingError(
            f'markitdown is not installed, so {_format_suffix(name)} files cannot be read; '
            f'install the readers extra: `pip install "plurai-simulation-sdk[readers]"` ({exc}).'
        ) from exc
    suffix = _format_suffix(name)
    converter = getattr(converters, _CONVERTERS[suffix])()
    return converter.convert(io.BytesIO(data), StreamInfo(extension=suffix, filename=_leaf(name))).text_content or ''


def pdf_reader() -> Literal['docling', 'markitdown']:
    """Which reader a corpus PDF gets: docling when plurai-extract's PDF extra is installed, else markitdown."""
    try:
        return 'docling' if importlib.util.find_spec('docling') is not None else 'markitdown'
    except (ImportError, ValueError):
        return 'markitdown'


def _render_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return str(value)


def _escape_structure(line: str) -> str:
    """A value line that markdown would read as a heading or a fence, escaped so the record stays one section."""
    return f'\\{line}' if line.lstrip().startswith(('#', '```', '~~~')) else line


def render_records(text: str) -> str:
    """JSON Lines as markdown: one ``# record <n>`` section per line, ``n`` its 1-based line number.

    Top-level headings, so the extractor's section chunker gives each record its own heading path rather
    than nesting every later record under the first.

    The body is one ``key: value`` line per field of an object, nested values as compact JSON, empty values
    left out; any other record is its compact JSON, and a line that is not JSON is kept verbatim. No field is
    taken for a title: the line number is the stable, citable name.
    """
    sections: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            body = line
        else:
            if isinstance(record, dict):
                body = '\n'.join(f'{key}: {_render_value(value)}' for key, value in record.items() if value not in (None, '', [], {}))
            else:
                body = _render_value(record)
        sections.append(f'# record {number}\n\n' + '\n'.join(_escape_structure(part) for part in body.splitlines()))
    return '\n\n'.join(sections) + '\n' if sections else ''


def _utf8(data: bytes) -> str | None:
    """Strict UTF-8 (a BOM dropped), or ``None``: a NUL also marks UTF-16 without a BOM."""
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return None
    return None if '\x00' in text else text


def to_text(name: str, raw: bytes) -> TextRead:
    """``name``'s text, read by its format: inflated when ``.gz``, then by :func:`kind_of` its inner name.

    ``text`` is strict UTF-8; ``records`` are :func:`render_records`; ``convert`` and ``pdf`` go through
    markitdown's converter for that format (a PDF that yields no text is ``no text extracted``). Never raises
    on one file's content: the reason is returned. Raises :class:`ConverterMissingError` only when
    markitdown itself is missing.
    """
    if isinstance(raw, Refused):
        return TextRead(None, raw.reason)
    try:
        name, data = decompress(name, raw)
    except UnreadableError as exc:
        return TextRead(None, str(exc))
    kind = kind_of(name)
    if kind is None:
        if _leaf(name).endswith(ZIP_SUFFIX):
            return TextRead(None, 'an archive: its members are read one by one')
        return TextRead(None, f'no reader for {_format_suffix(name) or "a file without a suffix"}')
    if kind in ('text', 'records'):
        text = _utf8(data)
        if text is None:
            return TextRead(None, 'not UTF-8 text')
        return TextRead(render_records(text) if kind == 'records' else text)
    try:
        text = _markitdown_convert(name, data)
    except ConverterMissingError:
        raise
    except Exception as exc:  # noqa: BLE001 — one file a converter cannot read is named, never the whole read
        return TextRead(None, f'could not be converted to text ({type(exc).__name__})')
    if kind == 'pdf' and not text.strip():
        return TextRead(None, 'no text extracted')
    return TextRead(text)


# --- the scan's reader ----------------------------------------------------------------------------------


class ScanReader:
    """The scan contract's ``SourceReader`` (``scan_contract.corpus``): :func:`members` to split a file into
    units, and a decode that reads converted formats as their text. A text format, inflated first when
    gzipped, keeps the scan's own ``decode_source`` (lossy decode, warned). Passed with
    :data:`SCAN_SUFFIXES` as the scope."""

    def members(self, path: str | os.PathLike[str]) -> Iterator[tuple[str, bytes]]:
        return members(path)

    def decode(self, name: str, raw: bytes) -> tuple[str, bool, str]:
        """``(text, undecodable, warning)`` for the unit ``name`` (``x.docx``, ``x.json.gz``, ``a.zip!/m.txt``)."""
        if isinstance(raw, Refused):
            return '', True, f'not read: {raw.reason}'
        leaf = name.rsplit('!/', 1)[-1]
        try:
            leaf, data = decompress(leaf, raw)
        except UnreadableError as exc:
            return '', True, f'not read: {exc}'
        if kind_of(leaf) in (None, 'text'):  # None: a suffix the operator admitted as plain text
            return decode_source(data)
        try:
            read = to_text(leaf, data)
        except ConverterMissingError as exc:
            return '', True, str(exc)
        if read.text is None:
            return '', True, f'not read: {read.reason}'
        return read.text, False, ''


SCAN_READER = ScanReader()
