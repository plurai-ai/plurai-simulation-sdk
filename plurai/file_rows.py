"""The world-state row for a file an agent produced or was handed: ``plurai.file_row``.

A harness whose agent writes files reports them from ``world_snapshot()``, one row per file, and the
judges read the row's ``text_preview`` as the file's content. Writing that reader by hand in every
``simulate.py`` is error-prone: a pptx loop that walks text frames only drops every table, and a
spreadsheet's uncomputed formulas read as ``NaN`` or as formula text a judge then "computes" itself.
The reader lives here instead, so every harness gets the same one:

* ``.xlsx`` / ``.xlsm`` / ``.xltx`` / ``.xltm``: each cell as the file stores it. A formula cell shows its
  formula text, and the value the file itself saved for it, labelled ``[stored value: ...]``; a formula
  cell the file saved no value for is named in ``_unevaluated_formulas`` (``'Sheet!A1'``). Nothing is
  calculated here: a value the file does not hold is never shown. Only the cells the file holds are
  read, row by row, and reading stops once the preview is full.
* ``.pptx``: every shape's text: text frames, table cells, and shapes nested in groups.
* plain-text suffixes: the file's text.
* ``.docx``: markitdown's reader for the body, led by the text of the page headers and footers, labelled
  ``Header:`` / ``Footer:`` (markitdown reads the body only, so a footer would otherwise not be shown).
* ``.pdf``: markitdown's reader for that format, which ships with the SDK.
* any other suffix: markitdown, which picks a reader from the file itself.

A preview carries at most :data:`PREVIEW_CHARS` characters. A longer one is cut there and the row says
so: ``_preview_truncated: {'shown_chars': ..., 'total_chars': ...}``, so a judge knows the rest of the
file exists and was not shown. ``total_chars`` is ``None`` when the reader stopped at the limit and
never read the rest (a spreadsheet), so its length is not known.

A produced image (``.png .jpg .jpeg .gif .webp``) also carries ``_image_path``, the file's absolute resolved
path, and ``image_media_type``, which comes from the file's first bytes (PNG, JPEG, GIF or WEBP), not its
suffix. A file with an image suffix whose bytes are none of those carries no ``_image_path`` and says so in
``_unreadable['image']``. The bytes never ride the row: the orchestrator uploads the file where it lives, with
the row's ``sha256`` as the checksum storage verifies, and replaces ``_image_path`` with a reference in
``image`` (``plurai.orchestrator.world_images``). :func:`without_images` takes both off the rows a per-turn
capture reads, which the simulated user sees as text.

A file whose suffix declares a binary format (the OOXML / zip family, ``.pdf``) but whose first bytes are not that
format carries ``_unreadable: {'content': 'not a .xlsx: its bytes are not that format'}``: the row's bytes are not the
document its name says.

A file no reader can render is ``text_preview: None`` plus ``_unreadable: {'text_preview':
'<ExcClass>: <message>'}``, never an error string in the content slot. That includes a file larger
than :data:`PREVIEW_MAX_BYTES` (``FileTooLargeToPreviewError``) and a spreadsheet read that runs past
:data:`XLSX_READ_SECONDS` (``PreviewBudgetExceededError``): a preview is never allowed to hang the run.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

#: How much of a file's text a row carries; a longer preview is cut here and marked.
PREVIEW_CHARS = 4000

#: A file larger than this is identified (size, sha256) but not previewed.
PREVIEW_MAX_BYTES = 64 * 1024 * 1024

#: Wall-clock seconds a spreadsheet read may take before it is abandoned as unreadable.
XLSX_READ_SECONDS = 10.0

#: How many uncomputed formula cells a row names; past it, ``_unevaluated_formulas_count`` has the total.
UNEVALUATED_LISTED = 100

#: The row key stating that ``text_preview`` was cut: ``{'shown_chars': int, 'total_chars': int | None}``.
PREVIEW_TRUNCATED_KEY = '_preview_truncated'

#: The row key naming each formula cell (``'Sheet!A1'``) the file saved no value for.
UNEVALUATED_FORMULAS_KEY = '_unevaluated_formulas'

#: The row key with the number of such cells, present when there are more than the row names.
UNEVALUATED_COUNT_KEY = '_unevaluated_formulas_count'

#: The row key naming each field whose reader failed, with the failure.
UNREADABLE_KEY = '_unreadable'

#: The row key holding a produced image's storage reference, set by the orchestrator once it uploads the file.
IMAGE_KEY = 'image'

#: The row key holding a produced image's absolute path until the orchestrator uploads it, and never after.
IMAGE_PATH_KEY = '_image_path'

#: The row key naming the media type of a produced image.
IMAGE_MEDIA_TYPE_KEY = 'image_media_type'

#: The image suffixes a row names a file for: the formats vision models accept. The file's own bytes decide the
#: row's ``image_media_type``, not the suffix.
IMAGE_SUFFIXES = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp'})

_TEXT_SUFFIXES = frozenset({'.txt', '.md', '.csv', '.tsv', '.json', '.html', '.xml', '.yaml', '.yml'})
_SPREADSHEET_SUFFIXES = frozenset({'.xlsx', '.xlsm', '.xltx', '.xltm'})
_HASH_CHUNK = 1024 * 1024
_SNIFF_BYTES = 12  # enough for every signature in :func:`_sniff_image` and :data:`_FORMAT_SIGNATURES`
#: The first bytes a binary document suffix declares: the OOXML / zip family is a zip archive, a PDF starts ``%PDF``.
#: A file under one of these suffixes whose bytes do not start so is not that format (a markdown note saved as
#: ``.xlsx``), and its row says so in ``_unreadable['content']``, as an image suffix does in ``_unreadable['image']``.
_ZIP_SIGNATURES = (b'PK\x03\x04', b'PK\x05\x06')  # a local file header; an empty archive's end record
_FORMAT_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    **dict.fromkeys(
        ('.xlsx', '.xlsm', '.xltx', '.xltm', '.docx', '.docm', '.dotx', '.dotm', '.pptx', '.pptm', '.potx', '.potm', '.zip'),
        _ZIP_SIGNATURES,
    ),
    '.pdf': (b'%PDF',),
}
_IMAGE_FIELDS = frozenset({IMAGE_KEY, IMAGE_PATH_KEY})


class FileTooLargeToPreviewError(Exception):
    """The file is larger than :data:`PREVIEW_MAX_BYTES`, so no reader is started on it."""


class PreviewBudgetExceededError(Exception):
    """A spreadsheet read ran past :data:`XLSX_READ_SECONDS` before the preview was full."""


class _Preview(NamedTuple):
    text: str
    unevaluated: list[str]  # at most UNEVALUATED_LISTED cells
    unevaluated_count: int = 0
    complete: bool = True  # False: the reader stopped at the preview limit and did not read the rest


_Reader = Callable[[Path], _Preview]


def file_row(path: str | os.PathLike[str], *, entity_type: str, preview: bool = True) -> dict[str, Any]:
    """The ``world_snapshot()`` row for one file: ``{_entity_type, name, suffix, size_bytes, sha256, ...}``.

    ``preview=True`` (a file the agent produced) adds ``text_preview``, at most :data:`PREVIEW_CHARS`
    characters (``_preview_truncated`` says how much was left out when it is cut), plus
    ``_unevaluated_formulas`` when a spreadsheet saved no value for some formula cells.
    ``preview=False`` (a file handed to the agent, identified but not graded on its content) adds
    ``content_type`` instead.
    """
    path = Path(path)
    digest = hashlib.sha256()
    head = b''  # the file's first bytes, which name its format
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        while chunk := stream.read(_HASH_CHUNK):
            digest.update(chunk)
            head = head or chunk[:_SNIFF_BYTES]
    row: dict[str, Any] = {
        '_entity_type': entity_type,
        'name': path.name,
        'suffix': path.suffix.lower(),
        'size_bytes': size,
        'sha256': digest.hexdigest(),
    }
    if (signatures := _FORMAT_SIGNATURES.get(row['suffix'])) is not None and not head.startswith(signatures):
        row[UNREADABLE_KEY] = {'content': f'not a {row["suffix"]}: its bytes are not that format'}
    if not preview:
        row['content_type'] = mimetypes.guess_type(path.name)[0]
        return row
    reader = _READERS.get(row['suffix'], _markitdown_text)
    try:
        if size > PREVIEW_MAX_BYTES:
            raise FileTooLargeToPreviewError(f'{size} bytes, over the {PREVIEW_MAX_BYTES}-byte preview limit')
        result = reader(path)
    except Exception as exc:  # a file the reader cannot open is typed, never handed out as its content
        row['text_preview'] = None
        row.setdefault(UNREADABLE_KEY, {})['text_preview'] = _unreadable_reason(exc)
    else:
        row['text_preview'] = result.text[:PREVIEW_CHARS]
        if not result.complete or len(result.text) > PREVIEW_CHARS:
            row[PREVIEW_TRUNCATED_KEY] = {'shown_chars': len(row['text_preview']), 'total_chars': len(result.text) if result.complete else None}
        if result.unevaluated:
            row[UNEVALUATED_FORMULAS_KEY] = result.unevaluated
            if result.unevaluated_count > len(result.unevaluated):
                row[UNEVALUATED_COUNT_KEY] = result.unevaluated_count
    if row['suffix'] in IMAGE_SUFFIXES:
        _add_image(row, path, head)
    return row


#: The import roots of the document readers the ``readers`` extra installs. None is a base dependency.
_READER_MODULES = frozenset({'markitdown', 'mammoth', 'pptx', 'openpyxl', 'pdfminer'})
_READERS_EXTRA_HINT = "install the document readers: pip install 'plurai-simulation-sdk[readers]'"


def _unreadable_reason(exc: Exception) -> str:
    """``exc`` as ``_unreadable['text_preview']`` says it, with the install line when a reader is not installed."""
    reason = f'{type(exc).__name__}: {exc}'
    # markitdown reports a format reader whose package is absent as its own exception, not as a failed import.
    missing_reader = type(exc).__name__ == 'MissingDependencyException' or (
        isinstance(exc, ModuleNotFoundError) and (exc.name or '').partition('.')[0] in _READER_MODULES
    )
    return f'{reason} ({_READERS_EXTRA_HINT})' if missing_reader else reason


def _add_image(row: dict[str, Any], path: Path, head: bytes) -> None:
    """Name the image file in the row: where it is and the type its bytes show, never the bytes themselves.

    The type comes from the file's first bytes, not its suffix: a JPEG saved as ``.png`` is ``image/jpeg``,
    and a file whose bytes are no supported image format is reported, not named for upload."""
    media_type = _sniff_image(head)
    if media_type is None:
        row.setdefault(UNREADABLE_KEY, {})[IMAGE_KEY] = f'not a {row["suffix"]} image: its bytes match no supported image format'
        return
    row[IMAGE_PATH_KEY] = str(path.resolve())
    row[IMAGE_MEDIA_TYPE_KEY] = media_type


def _sniff_image(head: bytes) -> str | None:
    """The media type of the image format ``head`` (a file's first bytes) starts as, or ``None``."""
    if head.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if head.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if head.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'image/webp'
    return None


def without_images(state: Mapping[str, Any]) -> dict[str, Any]:
    """``state`` with the ``image`` and ``_image_path`` fields off every produced-file row, input untouched.

    A produced-file row is one carrying ``_image_path`` or ``image_media_type``, which only :func:`file_row`
    writes. A domain row whose own field is named ``image`` (a product's picture URL) is not one, and passes whole.
    """
    return {
        entity_id: {key: value for key, value in row.items() if key not in _IMAGE_FIELDS}
        if isinstance(row, Mapping) and (IMAGE_PATH_KEY in row or IMAGE_MEDIA_TYPE_KEY in row)
        else row
        for entity_id, row in state.items()
    }


def _plain_text(path: Path) -> _Preview:
    return _Preview(path.read_text(encoding='utf-8', errors='replace'), [])


def _markitdown_text(path: Path) -> _Preview:
    from markitdown import MarkItDown

    return _Preview(MarkItDown().convert(str(path)).text_content, [])


def _markitdown_format(converter_name: str) -> _Reader:
    """One markitdown format reader and no other: a file that is not a valid ``.docx`` is unreadable,
    never re-read as plain text (markitdown's own fallback when its format reader fails)."""

    def read(path: Path) -> _Preview:
        from markitdown import StreamInfo, converters

        converter = getattr(converters, converter_name)()
        with path.open('rb') as stream:
            return _Preview(converter.convert(stream, StreamInfo(extension=path.suffix.lower(), filename=path.name)).text_content, [])

    return read


#: The WordprocessingML namespace, and the package parts that hold a document's page headers and footers.
_WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
_PAGE_FURNITURE_PART = re.compile(r'word/(header|footer)\d*\.xml')


def _docx_text(path: Path) -> _Preview:
    """The body as markitdown reads it, LED by the page headers and footers it does not read: short, and
    placed first so a long body cut at :data:`PREVIEW_CHARS` never cuts them away."""
    body = _markitdown_format('DocxConverter')(path).text
    furniture = _docx_page_furniture(path)
    return _Preview('\n'.join([*furniture, '', body]) if furniture else body, [])


def _docx_page_furniture(path: Path) -> list[str]:
    """One ``Header: ...`` / ``Footer: ...`` line per non-empty paragraph of each header and footer part, in part
    order, each distinct line once (a document repeats its default header on first and even pages)."""
    from xml.etree import ElementTree

    lines: list[str] = []
    with zipfile.ZipFile(path) as package:
        # Headers before footers, as on the page; within a kind, in part order.
        found = [(match.group(1), name) for name in package.namelist() if (match := _PAGE_FURNITURE_PART.fullmatch(name))]
        parts = sorted(found, key=lambda part: (part[0] != 'header', part[1]))
        for kind, name in parts:
            root = ElementTree.fromstring(package.read(name))  # noqa: S314 — the agent's own output, parsed by stdlib expat (no external entities)
            for paragraph in root.iter(f'{_WORD_NS}p'):
                text = ''.join(run.text or '' for run in paragraph.iter(f'{_WORD_NS}t')).strip()
                line = f'{kind.capitalize()}: {text}'
                if text and line not in lines:
                    lines.append(line)
    return lines


def _pptx_text(path: Path) -> _Preview:
    import pptx

    deck = pptx.Presentation(str(path))
    lines = []
    for number, slide in enumerate(deck.slides, 1):
        lines.append(f'## Slide {number}')
        lines.extend(_shape_text(shape) for shape in slide.shapes)
    return _Preview('\n'.join(line for line in lines if line), [])


def _shape_text(shape: Any) -> str:
    """A shape's text. ``has_table`` and ``has_text_frame`` are exclusive, and a group holds its own shapes."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    if getattr(shape, 'has_table', False):
        return '\n'.join(' | '.join(cell.text for cell in row.cells) for row in shape.table.rows)
    if getattr(shape, 'has_text_frame', False) and shape.text_frame.text:
        return shape.text_frame.text
    if getattr(shape, 'shape_type', None) == MSO_SHAPE_TYPE.GROUP:
        return '\n'.join(filter(None, (_shape_text(inner) for inner in shape.shapes)))
    return ''


def _xlsx_text(path: Path) -> _Preview:
    """Each cell the file holds, as stored: a value, or a formula with the value the file saved for it, if any.

    One pass over each sheet's XML, one row at a time, visiting only the cells the file holds (a stray
    cell far off at ``XFD1048576`` is one cell, not a rectangle). Reading stops once the text passes
    :data:`PREVIEW_CHARS`.
    """
    from openpyxl.reader.excel import ExcelReader
    from openpyxl.styles.stylesheet import apply_stylesheet

    deadline = time.monotonic() + XLSX_READ_SECONDS
    # openpyxl's own reader, minus its worksheet step: that step sizes every sheet up front, which
    # reads a sheet with no <dimension> element to its end before a single row is returned.
    reader = ExcelReader(path, read_only=True, data_only=True, keep_links=False)
    unevaluated: list[str] = []
    unevaluated_count = 0
    lines: list[str] = []
    size = -1  # the joined length: one newline fewer than lines
    try:
        reader.read_manifest()
        reader.read_strings()
        reader.read_workbook()
        apply_stylesheet(reader.archive, reader.wb)
        for sheet, rel in reader.parser.find_sheets():
            if rel.target not in reader.valid_files or 'chartsheet' in rel.Type:
                continue
            heading: str | None = f'## {sheet.name}'  # written with the sheet's first shown row
            with reader.archive.open(rel.target) as source:
                for row_number, cells in _stored_cells(source, reader).parse():
                    if time.monotonic() > deadline:
                        raise PreviewBudgetExceededError(f'read stopped after {XLSX_READ_SECONDS:g}s at {sheet.name}!row {row_number}')
                    if all(cell['value'] is None and cell.get('formula') is None for cell in cells):
                        continue
                    rendered: list[str] = []
                    for cell in cells:
                        rendered.extend([''] * (cell['column'] - len(rendered) - 1))  # the gap before this cell
                        text, uncomputed = _cell_text(cell)
                        rendered.append(text)
                        if uncomputed:
                            unevaluated_count += 1
                            if len(unevaluated) < UNEVALUATED_LISTED:
                                unevaluated.append(f'{sheet.name}!{_coordinate(cell)}')
                    for line in (heading, ' | '.join(rendered)):
                        if line is not None:
                            lines.append(line)
                            size += len(line) + 1
                    heading = None
                    if size > PREVIEW_CHARS:
                        return _Preview('\n'.join(lines), unevaluated, unevaluated_count, complete=False)
    finally:
        reader.archive.close()
    return _Preview('\n'.join(lines), unevaluated, unevaluated_count)


def _stored_cells(source: Any, reader: Any) -> Any:
    """openpyxl's own worksheet parser, reporting each cell's stored value AND its formula in one pass."""
    from openpyxl.worksheet._reader import FORMULA_TAG, VALUE_TAG, WorkSheetParser

    class _StoredCellParser(WorkSheetParser):
        def parse_cell(self, element: Any) -> dict[str, Any]:
            cell = super().parse_cell(element)  # data_only: the value the file saved
            if element.find(FORMULA_TAG) is not None:
                if cell['value'] is None and element.get('t') == 'str' and element.find(VALUE_TAG) is not None:
                    cell['value'] = ''  # a formula whose saved result is the empty string
                self.data_only = False
                try:
                    cell['formula'] = self.parse_formula(element)
                finally:
                    self.data_only = True
            return cell

    book = reader.wb
    return _StoredCellParser(
        source, reader.shared_strings, data_only=True, epoch=book.epoch, date_formats=book._date_formats, timedelta_formats=book._timedelta_formats
    )


def _cell_text(cell: dict[str, Any]) -> tuple[str, bool]:
    """A cell's preview text, and whether it is a formula the file saved no value for."""
    value, formula = cell['value'], cell.get('formula')
    if formula is None:
        return ('' if value is None else str(value)), False
    if isinstance(formula, str):
        text = formula
    elif getattr(formula, 'text', None):  # an array (CSE / dynamic-array) formula
        text = formula.text
    else:  # a what-if data table
        text = f'=TABLE({getattr(formula, "r1", None) or ""},{getattr(formula, "r2", None) or ""})'
    if value is None:
        return text, True
    return f'{text} [stored value: {value}]', False


def _coordinate(cell: dict[str, Any]) -> str:
    from openpyxl.utils import get_column_letter

    return f'{get_column_letter(cell["column"])}{cell["row"]}'


#: Which reader previews which suffix; any other suffix goes to markitdown, which picks a reader itself.
_READERS: dict[str, _Reader] = {
    **dict.fromkeys(_SPREADSHEET_SUFFIXES, _xlsx_text),
    '.pptx': _pptx_text,
    '.docx': _docx_text,
    '.pdf': _markitdown_format('PdfConverter'),
    **dict.fromkeys(_TEXT_SUFFIXES, _plain_text),
}
